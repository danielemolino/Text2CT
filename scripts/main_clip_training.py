import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from monai.data import DataLoader
from monai.transforms import Compose

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from clip_training.utils.util import mkdir, set_seed, load_config_file  # noqa: E402
from clip_training.Clip_Training_Script import train  # noqa: E402
from core.cfg_helper import model_cfg_bank  # noqa: E402
from core.models.common.get_model import get_model  # noqa: E402
from clip_training.utils.logger import setup_logger  # noqa: E402

TRAINER_CONFIG_PATH = ROOT / "clip_training/clip_train_config.yaml"


def load_filenames(data_list_path: str) -> list:
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]
    return [_item["image"] for _item in filenames_train]


def prepare_data(train_files: list, reports_csv: str, cache_rate: float, num_workers: int = 2, batch_size: int = 1) -> DataLoader:
    reports = pd.read_csv(reports_csv)
    volume_text_mapping = {
        row["VolumeName"]: f"Findings: {row['Findings_EN']} Impression: {row['Impressions_EN']}"
        for _, row in reports.iterrows()
    }

    def lookup_text(volume_name: str) -> str:
        return volume_text_mapping.get(volume_name, "")

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(keys="impression", func=lookup_text),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    return DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    config = load_config_file(TRAINER_CONFIG_PATH)

    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    filename = f"clip_training_logs_{config.name}.txt"
    logger = setup_logger("CLIP TRAINING", config.logs, 0, filename=filename)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    set_seed(seed=getattr(config, "seed", 11), n_gpu=config.n_gpu)

    cfgm = model_cfg_bank()(config.clip_model)
    clip = get_model()(cfgm)
    clip = clip.to(config.device)

    logger.info(f"Training parameters: {config}")

    filenames_train = load_filenames(config.data_list)
    train_files = []
    for image_path in filenames_train:
        if not os.path.exists(image_path):
            continue
        volume_name = os.path.basename(image_path)
        train_files.append({"image": image_path, "impression": volume_name})

    dataloader = prepare_data(
        train_files,
        reports_csv=config.reports_csv,
        cache_rate=0,
        batch_size=config.per_gpu_train_batch_size,
        num_workers=config.num_workers,
    )

    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    mkdir(config.checkpoint_dir)
    global_step, avg_loss = train(config, dataloader, clip, logger)

    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)


if __name__ == "__main__":
    main()
