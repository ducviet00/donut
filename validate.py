import json
import re

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from loguru import logger
from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from config import settings
from utils import JSONParseEvaluator


def validate(model: VisionEncoderDecoderModel, processor: DonutProcessor, dataset_subset: str, device = "cuda" if torch.cuda.is_available() else "cpu"):


    model.eval()
    model.to(device)

    output_list = []
    ground_truths = []
    accs = []

    dataset = load_dataset("naver-clova-ix/cord-v2", split=dataset_subset)
    evaluator = JSONParseEvaluator()

    for _, sample in tqdm(enumerate(dataset), total=len(dataset)):
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

        score = evaluator.cal_acc(seq, ground_truth)

        accs.append(score)
        output_list.append(seq)
        ground_truths.append(ground_truth)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs), "f1_accuracy": evaluator.cal_f1(output_list, ground_truths),}
    
    return scores

if __name__ == "__main__":
    config = VisionEncoderDecoderConfig.from_pretrained("logs/cord-19062023-20:12:00")
    config.encoder.image_size = settings.image_size  # (height, width)
    config.decoder.max_length = settings.max_length
    model = VisionEncoderDecoderModel.from_pretrained(
        "logs/cord-19062023-20:12:00", config=config
    )
    processor = DonutProcessor.from_pretrained("logs/cord-19062023-20:12:00")
    scores = validate(model=model, processor=processor, dataset_subset="validation")
    logger.info(f"Mean validation set accuracy: {scores['mean_accuracy']}")
    logger.info(f"Mean validation set F1 Score: {scores['f1_accuracy']}")
    scores = validate(model=model, processor=processor, dataset_subset="test")
    logger.info(f"Mean test set accuracy: {scores['mean_accuracy']}")
    logger.info(f"Mean test set F1 Score: {scores['f1_accuracy']}")