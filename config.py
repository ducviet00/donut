from pydantic import BaseSettings
from datetime import datetime

class Settings(BaseSettings):
    dataset_name: str = "naver-clova-ix/cord-v2"
    task_start_token: str = "<s_cord-v2>"
    prompt_end_token: str = "<s_cord-v2>"
    image_size: list = [1280, 960]
    max_length: int = 768
    max_epochs: int = 30
    val_check_interval: float = 0.2
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    num_training_samples_per_epoch: int = 800
    lr: float = 3e-5
    train_batch_size: int = 1
    val_batch_size: int = 1
    seed: int = 2023
    num_nodes: int = 1
    warmup_steps: int = 300
    result_path: str = "./result"
    verbose: bool = True
    gpu_devices = 1
    num_sanity_val_steps = -1
    log_name = f"cord-{datetime.now().strftime('%d%m%Y-%H:%M:%S')}"

settings = Settings()

