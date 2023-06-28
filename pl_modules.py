import math
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import evaluate

from config import settings


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, processor, model, train_dataloader, val_dataloader):
        super().__init__()
        self.processor = processor
        self.model = model
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=settings.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )


        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if settings.verbose is True and len(scores) == 1:
                logger.info(f"Prediction: {pred}")
                logger.info(f"    Answer: {answer}")
                logger.info(f" Normed ED: {scores[0]}")

        val_edit_distance = np.mean(scores)
        self.log("val_edit_distance", val_edit_distance, sync_dist=True, batch_size=settings.val_batch_size)

        cer = evaluate.load("cer")
        cer.add_batch(predictions=predictions, references=answers)
        cer_score = cer.compute()
        self.log("character_error_rate", cer_score, sync_dist=True, batch_size=settings.val_batch_size)

        return (val_edit_distance, cer_score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=settings.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, settings.max_steps, settings.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        self.model.save_pretrained(f"{self.logger.save_dir}/{settings.log_name}")