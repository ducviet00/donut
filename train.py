import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from loguru import logger
from nltk import edit_distance
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from config import settings
from data import DonutDataset, SynthdogDataset, CORDOCRDataset
from donut import DonutModel
from pl_modules import DonutModelPLModule
from utils import JSONParseEvaluator
from validate import validate

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.makedirs(f"logs/{settings.log_name}/", exist_ok=True)
logger.add(f"logs/{settings.log_name}/train.log")


torch.set_float32_matmul_precision("medium")
def get_data(model: VisionEncoderDecoderModel, processor: DonutProcessor):
    if settings.pre_training:
        train_dataset = SynthdogDataset(
            model=model,
            processor=processor,
            dataset_name_or_path=settings.dataset_name,
            max_length=settings.max_length,
            split="train",
            task_start_token=settings.task_start_token,
            prompt_end_token=settings.prompt_end_token,
            sample_size=1000
        )
        if settings.cord_ocr_path:
            train_cord_dataset = CORDOCRDataset(
                model=model,
                processor=processor,
                dataset_name_or_path=settings.cord_ocr_path,
                max_length=settings.max_length,
                split="train",
                task_start_token=settings.task_start_token,
                prompt_end_token=settings.prompt_end_token,
            )
            train_dataset = ConcatDataset([train_dataset, train_cord_dataset])

        val_dataset = SynthdogDataset(
            model=model,
            processor=processor,
            dataset_name_or_path=settings.dataset_name,
            max_length=settings.max_length,
            split="validation",
            task_start_token=settings.task_start_token,
            prompt_end_token=settings.prompt_end_token,
            sample_size=500
        )
    else:
        train_dataset = DonutDataset(
            model=model,
            processor=processor,
            dataset_name_or_path=settings.dataset_name,
            max_length=settings.max_length,
            split="train",
            task_start_token=settings.task_start_token,
            prompt_end_token=settings.prompt_end_token,
            sort_json_key=False
        )

        val_dataset = DonutDataset(
            model=model,
            processor=processor,
            dataset_name_or_path=settings.dataset_name,
            max_length=settings.max_length,
            split="validation",
            task_start_token=settings.task_start_token,
            prompt_end_token=settings.prompt_end_token,
            sort_json_key=False
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=settings.train_batch_size, shuffle=True, num_workers=24
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=settings.val_batch_size, shuffle=False, num_workers=24
    )

    return train_dataloader, val_dataloader


def main():
    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained(settings.base_model)
    config.encoder.image_size = settings.image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = settings.max_length
    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
    if False:
        model = DonutModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path="microsoft/swin-base-patch4-window12-384",
            decoder_pretrained_model_name_or_path="hyunwoongko/asian-bart-ecjk",
            base_config=config,
        )
    else:
        model = VisionEncoderDecoderModel.from_pretrained(
            settings.base_model, config=config
        )

    processor = DonutProcessor.from_pretrained(settings.base_model)
    processor.image_processor.size = settings.image_size[
        ::-1
    ]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    train_dataloader, val_dataloader = get_data(model=model, processor=processor)

    batch = next(iter(train_dataloader))
    pixel_values, labels, target_sequences = batch

    logger.info(
        f"""
        Verify batch
        Shape of pixel_values: {pixel_values.shape}
        First 100 labels: {labels[:100]}
        Target Sequences: {target_sequences[:100]}
        """
    )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        [settings.task_start_token]
    )[0]
    # sanity check
    logger.info(f"Pad token ID: {processor.decode([model.config.pad_token_id])}")
    logger.info(
        f"Decoder start token ID: {processor.decode([model.config.decoder_start_token_id])}"
    )

    tensorboard_logger = TensorBoardLogger(save_dir="logs", name=settings.log_name)
    if settings.pre_training:
        early_stop_callback = EarlyStopping(
            monitor="train_loss", patience=10, verbose=True, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='character_error_rate'
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor="val_edit_distance", patience=10, verbose=True, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_edit_distance'
        )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=settings.gpu_devices,
        max_epochs=settings.max_epochs,
        max_steps=settings.max_steps,
        val_check_interval=settings.val_check_interval,
        check_val_every_n_epoch=settings.check_val_every_n_epoch,
        gradient_clip_val=settings.gradient_clip_val,
        precision="bf16-mixed",  # we'll use mixed precision
        num_sanity_val_steps=settings.num_sanity_val_steps,
        logger=tensorboard_logger,
        log_every_n_steps=settings.log_every_n_steps,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    model_module = DonutModelPLModule(
        processor, model, train_dataloader, val_dataloader
    )
    trainer.fit(model_module)

    if not settings.pre_training:
        scores = validate(model=model, processor=processor, dataset_subset="validation")
        logger.info(f"Mean accuracy: {scores['mean_accuracy']}")

    model_module.processor.save_pretrained(f"logs/{settings.log_name}")
    model_module.model.save_pretrained(f"logs/{settings.log_name}")


if __name__ == "__main__":
    main()
