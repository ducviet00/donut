import os
from datetime import datetime
from pydantic import BaseSettings

pre_training = True


class Settings(BaseSettings):
    dataset_name: str = "naver-clova-ix/cord-v2"
    task_start_token: str = "<s_cord-v2>"
    prompt_end_token: str = "<s_cord-v2>"
    image_size: list = [1280, 960]
    max_length: int = 768
    max_epochs: int = -1
    max_steps: int = 10000
    val_check_interval: float = 1
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 10
    lr: float = 3e-5
    train_batch_size: int = 1
    val_batch_size: int = 1
    seed: int = 2023
    num_nodes: int = 1
    warmup_steps: int = 100
    verbose: bool = False
    gpu_devices = 1
    num_sanity_val_steps = 2
    pre_training = False
    log_name = f"cord-{datetime.now().strftime('%d%m%Y-%H:%M:%S')}"


if Settings().pre_training:
    settings = Settings(
        dataset_name="naver-clova-ix/synthdog-en",
        task_start_token="<s_synthdog>",
        prompt_end_token="<s_synthdog>",
        image_size=[2048, 1536],
        max_length=1024,
        val_check_interval=1,
        gpu_devices=8,
        lr=1e-4,
        train_batch_size=2,
        val_batch_size=12,
        pre_training=True,
        verbose=True
    )
else:
    settings = Settings()
