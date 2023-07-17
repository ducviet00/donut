import json
import os
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

added_tokens = []

class CORDOCRDataset(Dataset):
    """
    PyTorch Dataset for Synthdog. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path (png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into pixel_values (vectorized image) and labels (input_ids of the tokenized string).

    Args:
        dataset_name_or_path: name of dataset the path containing image files
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        processor: DonutProcessor,
        model: VisionEncoderDecoderModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
    ):
        super().__init__()

        self.processor = processor
        self.model = model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token

        self.dataset_path = dataset_name_or_path
        self.gt_token_sequences = []
        self.images = []
        self.prepare_labels()
        self.dataset_length = len(self.images)

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def prepare_labels(self):
        for _name in os.listdir(self.dataset_path):
            if not _name.endswith("json"):
                continue
            _id = _name.split('.')[0]
            with open(os.path.join(self.dataset_path, f"{_id}.json"), "r") as f:
                ground_truth = json.load(f)
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            self.gt_token_sequences.append(ground_truth["gt_parse"]["text_sequence"] + self.processor.tokenizer.eos_token)
            self.images.append(Image.open(os.path.join(self.dataset_path, f"{_id}.png")))
    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        image = self.images[idx]

        # inputs
        pixel_values = self.processor(image, random_padding=self.split == "train", return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        target_sequence = self.gt_token_sequences[idx]
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return pixel_values, labels, target_sequence