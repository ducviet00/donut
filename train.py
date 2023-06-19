import json
import math
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from loguru import logger
from nltk import edit_distance
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (DonutProcessor, VisionEncoderDecoderConfig,
                          VisionEncoderDecoderModel)

from config import settings
from data.cord import DonutDataset
from donut import JSONParseEvaluator
from pl_modules import DonutModelPLModule


def validate(model: VisionEncoderDecoderModel, processor: DonutProcessor):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    output_list = []
    accs = []

    dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare encoder inputs
        pixel_values = processor(
            sample["image"].convert("RGB"), return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(device)
        # prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(device)

        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        seq = re.sub(
            r"<.*?>", "", seq, count=1
        ).strip()  # remove first task start token
        seq = processor.token2json(seq)

        ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        evaluator = JSONParseEvaluator()
        score = evaluator.cal_acc(seq, ground_truth)

        accs.append(score)
        output_list.append(seq)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    
    return scores

def get_data(model: VisionEncoderDecoderModel, processor: DonutProcessor):
    train_dataset = DonutDataset(
        model=model,
        processor=processor,
        dataset_name_or_path=settings.dataset_name,
        max_length=settings.max_length,
        split="train",
        task_start_token=settings.task_start_token,
        prompt_end_token=settings.prompt_end_token,
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )

    val_dataset = DonutDataset(
        model=model,
        processor=processor,
        dataset_name_or_path=settings.dataset_name,
        max_length=settings.max_length,
        split="validation",
        task_start_token=settings.task_start_token,
        prompt_end_token=settings.prompt_end_token,
        sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    )
    # feel free to increase the batch size if you have a lot of memory
    # I'm fine-tuning on Colab and given the large image size, batch size > 1 is not feasible
    train_dataloader = DataLoader(
        train_dataset, batch_size=settings.train_batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(val_dataset, batch_size=settings.val_batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader


def main():
    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = settings.image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = settings.max_length
    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602

    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base", config=config
    )
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
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
        First 100 labels: {labels}
        Target Sequences: {target_sequences}
        """
    )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([settings.task_start_token])[0]
    # sanity check
    logger.info("Pad token ID:", processor.decode([model.config.pad_token_id]))
    logger.info("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

    csv_logger = CSVLogger(save_dir=".", name="demo-run-cord")

    early_stop_callback = EarlyStopping(
        monitor="val_edit_distance", patience=3, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=settings.gpu_devices,
        max_epochs=settings.max_epochs,
        val_check_interval=settings.val_check_interval,
        check_val_every_n_epoch=settings.check_val_every_n_epoch,
        gradient_clip_val=settings.gradient_clip_val,
        precision=16,  # we'll use mixed precision
        num_sanity_val_steps=settings.num_sanity_val_steps,
        logger=csv_logger,
        callbacks=[early_stop_callback],
    )

    model_module = DonutModelPLModule(processor, model, train_dataloader, val_dataloader)
    trainer.fit(model_module)
    scores = validate(model=model, processor=processor)
    logger.info("Mean accuracy:", scores["mean_accuracy"])

if __name__ == "__main__":
    main()