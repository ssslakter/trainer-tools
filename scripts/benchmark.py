#!/usr/bin/env python
"""
Benchmark: Single-GPU vs DataParallel vs Accelerate (DDP)
=========================================================

Compares training throughput (samples/sec) across backends.

Usage::

    # Run everything (single-GPU, DP, then spawns DDP automatically):
    python benchmark.py

    # DDP-only mode (called internally by the script, or via accelerate launch):
    accelerate launch --num_processes=2 benchmark.py --mode accelerate

    # Tune parameters:
    python benchmark.py --epochs 5 --batch-size 128 --num-samples 8192
"""

import argparse
import logging
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trainer_tools.trainer import Trainer
from trainer_tools.hooks import MetricsHook, ProgressBarHook, Loss
from trainer_tools.hooks.metrics import SamplesPerSecond
from trainer_tools.hooks.accelerate import AccelerateHook

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic dataset & model
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Random image-classification dataset (pre-generated in RAM)."""

    def __init__(self, num_samples: int = 4096, img_size: int = 32, num_classes: int = 10):
        self.x = torch.randn(num_samples, 3, img_size, img_size)
        self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loaders(batch_size: int, num_samples: int):
    ds = SyntheticDataset(num_samples=num_samples)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
    return train_dl, valid_dl


def _make_metrics(batch_size):
    return MetricsHook(
        metrics=[Loss(), SamplesPerSecond(batch_size=batch_size)],
        history_file="/dev/null",
        verbose=True,
    )


def bench(label, run_fn, **kwargs):
    """Time a training run and return (label, elapsed, samples/sec)."""
    t0 = time.perf_counter()
    total_samples = run_fn(**kwargs)
    elapsed = time.perf_counter() - t0
    sps = total_samples / elapsed
    return label, elapsed, sps


# ---------------------------------------------------------------------------
# Training runners
# ---------------------------------------------------------------------------

def run_single(device, train_dl, valid_dl, epochs, batch_size, num_samples):
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model, train_dl=train_dl, valid_dl=valid_dl, optim=opt,
        loss_func=nn.CrossEntropyLoss(), epochs=epochs,
        hooks=[_make_metrics(batch_size), ProgressBarHook()],
        device=device,
    )
    trainer.fit()
    return num_samples * epochs


def run_data_parallel(device, train_dl, valid_dl, epochs, batch_size, num_samples):
    model = nn.DataParallel(SmallCNN()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model, train_dl=train_dl, valid_dl=valid_dl, optim=opt,
        loss_func=nn.CrossEntropyLoss(), epochs=epochs,
        hooks=[_make_metrics(batch_size), ProgressBarHook()],
        device=device,
    )
    trainer.fit()
    return num_samples * epochs


def run_accelerate(train_dl, valid_dl, epochs, batch_size, num_samples):
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model, train_dl=train_dl, valid_dl=valid_dl, optim=opt,
        loss_func=nn.CrossEntropyLoss(), epochs=epochs,
        hooks=[AccelerateHook(), _make_metrics(batch_size), ProgressBarHook()],
        device="cpu",  # AccelerateHook overrides
    )
    trainer.fit()
    return num_samples * epochs


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_header(title):
    log.info("\n" + "=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def print_summary(results):
    log.info("\n" + "=" * 60)
    log.info("  Summary")
    log.info("=" * 60)
    log.info("  %-25s %10s %14s", "Backend", "Time (s)", "Samples/sec")
    log.info("  " + "-" * 51)
    for name, elapsed, sps in results:
        log.info("  %-25s %10.1f %14.0f", name, elapsed, sps)
    log.info("")


# ---------------------------------------------------------------------------
# Mode: accelerate-only (called via `accelerate launch`)
# ---------------------------------------------------------------------------

def mode_accelerate(args):
    """Runs only the Accelerate benchmark. Meant to be invoked via accelerate launch."""
    train_dl, valid_dl = make_loaders(args.batch_size, args.num_samples)

    from accelerate import PartialState
    state = PartialState()
    is_main = state.is_main_process
    num_procs = state.num_processes

    if is_main:
        print_header(f"AccelerateHook  (DDP, {num_procs} processes)")

    t0 = time.perf_counter()
    run_accelerate(train_dl, valid_dl, args.epochs, args.batch_size, args.num_samples)
    elapsed = time.perf_counter() - t0
    sps = args.num_samples * args.epochs / elapsed

    if is_main:
        log.info("  %.1f sec | %.0f samples/sec", elapsed, sps)


# ---------------------------------------------------------------------------
# Mode: full comparison (single-process entry point)
# ---------------------------------------------------------------------------

def mode_full(args):
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    train_dl, valid_dl = make_loaders(args.batch_size, args.num_samples)
    common = dict(train_dl=train_dl, valid_dl=valid_dl, epochs=args.epochs,
                  batch_size=args.batch_size, num_samples=args.num_samples)
    results = []

    # --- Single device -------------------------------------------------------
    print_header(f"Single device  ({device})")
    results.append(bench("Single device", run_single, device=device, **common))
    log.info("  %.1f sec | %.0f samples/sec", results[-1][1], results[-1][2])

    # --- DataParallel --------------------------------------------------------
    if has_multi_gpu:
        print_header(f"DataParallel  ({num_gpus} GPUs)")
        results.append(bench("DataParallel", run_data_parallel, device=device, **common))
        log.info("  %.1f sec | %.0f samples/sec", results[-1][1], results[-1][2])
    else:
        log.info("\n(Skipping DataParallel — need ≥2 GPUs)")

    # --- Accelerate (single-process) -----------------------------------------
    print_header("AccelerateHook  (single process)")
    results.append(bench("Accelerate (1 proc)", run_accelerate, **common))
    log.info("  %.1f sec | %.0f samples/sec", results[-1][1], results[-1][2])

    # --- Accelerate DDP (multi-process, spawned) -----------------------------
    if has_multi_gpu:
        print_header(f"AccelerateHook  (DDP, {num_gpus} GPUs) — spawning")
        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            "--num_processes", str(num_gpus),
            __file__,
            "--mode", "accelerate",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-samples", str(args.num_samples),
        ]
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - t0
        sps = args.num_samples * args.epochs / elapsed
        results.append((f"Accelerate DDP ({num_gpus} GPUs)", elapsed, sps))
    else:
        log.info("\n(Skipping DDP — need ≥2 GPUs)")

    print_summary(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark trainer backends")
    parser.add_argument("--mode", choices=["full", "accelerate"], default="full",
                        help="'full' = run all backends; 'accelerate' = DDP-only (for accelerate launch)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None, help="Force device (default: auto)")
    args = parser.parse_args()

    if args.mode == "accelerate":
        mode_accelerate(args)
    else:
        mode_full(args)


if __name__ == "__main__":
    main()
