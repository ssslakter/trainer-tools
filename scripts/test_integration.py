#!/usr/bin/env python
"""
Test integration of AccelerateHook, CheckpointHook, and ProgressBarHook.
Now with increased workload to visualize progress bar increments.

Usage::
    accelerate launch scripts/test_integration.py
"""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from trainer_tools.trainer import Trainer
from trainer_tools.hooks import CheckpointHook, ProgressBarHook
from trainer_tools.hooks.accelerate import AccelerateHook


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Larger random image-classification dataset to slow things down."""

    def __init__(self, num_samples: int = 1024, img_size: int = 128, num_classes: int = 10):
        # Increased img_size from 16 to 128 to increase compute load
        self.x = torch.randn(num_samples, 3, img_size, img_size)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class HeavierCNN(nn.Module):
    """A slightly wider CNN to ensure the GPU/CPU has real work to do."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def simple_train_step(batch, trainer):
    # Artificial delay to make the progress bar "breathe"
    # time.sleep(0.02)
    x, y = batch
    out = trainer.model(x)
    loss = nn.functional.cross_entropy(out, y)
    return {"loss": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="tmp_integration_checkpoints")
    args = parser.parse_args()

    ds = SyntheticDataset(num_samples=args.num_samples, img_size=128)
    train_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = HeavierCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    accelerate_hook = AccelerateHook(gradient_accumulation_steps=args.grad_accum)

    ckpt_hook = CheckpointHook(
        save_dir=str("temp"),
        save_every_steps=args.save_every,
        keep_last=5,
    )
    pbar_hook = ProgressBarHook(freq=1)

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=None,
        optim=opt,
        train_step=simple_train_step,
        epochs=args.epochs,
        hooks=[accelerate_hook, ckpt_hook, pbar_hook],
    )

    log.info("Starting heavy training...")
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
