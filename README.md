[![PyPI version](https://img.shields.io/pypi/v/trainer-tools.svg)](https://pypi.org/project/trainer-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub Repo stars](https://img.shields.io/github/stars/ssslakter/trainer-tools?style=social)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trainer-tools)
# Trainer Tools

A lightweight, hook-based training loop for PyTorch. `trainer-tools` abstracts away the boilerplate of training loops while remaining fully customizable via a powerful flexible hook system.

## Features

*   **Hook System**: Customize every step of the training lifecycle (before/after batch, step, epoch, fit).
*   **Built-in Integrations**: Comes with hooks for wandb or trackio, Progress Bar, and Checkpointing.
*   **Optimization**: Easy Automatic Mixed Precision (AMP), Gradient Accumulation, and Gradient Clipping.
*   **Metrics**: robust metric tracking and logging to JSONL or external trackers.
*   **Memory Profiling**: Built-in tools to debug CUDA memory leaks.

## Installation

```bash
pip install trainer-tools

# With optional integrations
pip install trainer-tools[wandb]      # Weights & Biases logging
pip install trainer-tools[trackio]    # Trackio logging
pip install trainer-tools[hydra]      # Hydra config management
pip install trainer-tools[all]        # All optional dependencies
```

## Quick Start

Here is a minimal example of training a simple model:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import MetricsHook, Accuracy, Loss, ProgressBarHook

# 1. Prepare Data
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
ds = TensorDataset(x, y)
dl = DataLoader(ds, batch_size=32)

# 2. Define Model
model = nn.Sequential(nn.Linear(10, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. Setup Hooks
metrics = MetricsHook(metrics=[Accuracy(), Loss()])
pbar = ProgressBarHook()

# 4. Train
trainer = Trainer(
    model=model,
    train_dl=dl,
    valid_dl=dl,
    optim=optimizer,
    loss_func=nn.CrossEntropyLoss(),
    epochs=5,
    hooks=[metrics, pbar],
    device="cuda" if torch.cuda.is_available() else "cpu"
)

trainer.fit()
```

## How the Trainer Works

### Training Loop

`Trainer.fit()` runs a standard PyTorch training loop but exposes **hook points** at every meaningful stage so you can inject logic without touching the core loop:

```
fit()
├── before_fit
└── for each epoch:
    ├── before_epoch
    ├── for each training batch:
    │   ├── before_step
    │   ├── predict()        → trainer.preds
    │   ├── after_pred
    │   ├── get_loss()       → trainer.loss / trainer.loss_t # float loss / tensor loss
    │   ├── after_loss
    │   ├── loss_t.backward()
    │   ├── after_backward
    │   ├── opt.step() / opt.zero_grad()
    │   └── after_step
    ├── before_valid
    ├── for each validation batch:
    │   └── (same as above, no backward/opt steps)
    └── after_epoch
└── after_fit           (or after_cancel on KeyboardInterrupt)
```

### Key Trainer Attributes

At any hook point you have access to the live trainer state:

| Attribute | Description |
|---|---|
| `trainer.model` | The `nn.Module` being trained |
| `trainer.opt` | The optimizer |
| `trainer.epoch` / `trainer.step` | Current epoch / global step count |
| `trainer.batch` | The raw batch from the dataloader |
| `trainer.preds` | Model predictions (set after `predict()`) |
| `trainer.loss` / `trainer.loss_t` | Scalar loss value / loss tensor |
| `trainer.training` | `True` during the train phase, `False` during validation |
| `trainer.dl` | DataLoader currently in use |
| `trainer.config` | Optional config object (e.g. Hydra `DictConfig`) |

Three boolean flags let a hook short-circuit the default behavior for a single step:

| Flag | Effect when set to `True` |
|---|---|
| `trainer.skip_backward` | Skips `loss.backward()` |
| `trainer.skip_opt_step` | Skips `opt.step()` |
| `trainer.skip_zero_grad` | Skips `opt.zero_grad()` |

### Hook Execution Order

Hooks are sorted by their `ord` attribute before each call. Lower values run first. The default is `0`. This guarantees correct ordering when hooks depend on one another (e.g., `CheckpointHook` runs before `LRSchedulerHook` so a restored scheduler state is used from the beginning).

## The Hook System

`trainer-tools` relies on `BaseHook`. You can create custom behavior by subclassing it:

```python
from trainer_tools.hooks import BaseHook

class MyCustomHook(BaseHook):
    def after_step(self, trainer):
        if trainer.step % 100 == 0:
            print(f"Current Loss: {trainer.loss}")
```

### Available Hooks

#### `ProgressBarHook`
Displays `tqdm` progress bars for epochs and batches. Shows a running training loss that updates every `freq` steps.

```python
from trainer_tools.hooks import ProgressBarHook
pbar = ProgressBarHook(freq=10)
```

#### `MetricsHook`
Central hub for computing, aggregating, and logging metrics. Supports logging to the console, a JSONL history file, **Weights & Biases**, or **Trackio**.

```python
from trainer_tools.hooks import MetricsHook, Loss
from trainer_tools.hooks.metrics import Accuracy

metrics = MetricsHook(
    metrics=[Loss(), Accuracy()],
    tracker_type="wandb",   # or "trackio" / None
    project="my-project",
    name="run_1"
)
```

Metrics are split into phases (`"step"` or `"epoch"`) and automatically prefixed with `train_` / `valid_`.

#### `LRSchedulerHook`
Wraps any PyTorch `LRScheduler` and calls `sched.step()` after every optimizer update.

```python
from trainer_tools.hooks import LRSchedulerHook

hook = LRSchedulerHook(
    sched_fn=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
)
```

#### `AMPHook`
Enables Automatic Mixed Precision training by wrapping the forward pass in `torch.autocast` and managing a `GradScaler`. Supports both `float16` and `bfloat16`.

```python
from trainer_tools.hooks import AMPHook
amp = AMPHook(dtype=torch.bfloat16, device_type="cuda")
```

#### `GradClipHook`
Clips gradient norms before each optimizer step to stabilize training.

```python
from trainer_tools.hooks import GradClipHook
clip = GradClipHook(max_norm=1.0)
```

#### `GradientAccumulationHook`
Accumulates gradients over multiple micro-batches before calling `opt.step()`, effectively increasing the batch size without extra GPU memory.

```python
from trainer_tools.hooks import GradientAccumulationHook
accum = GradientAccumulationHook(accumulation_steps=4)
```

#### `CheckpointHook`
Saves and restores model, optimizer, scheduler, scaler, and RNG states. Supports both a `"best"` (by a tracked metric) and `"latest"` save strategy, and keeps a configurable number of recent checkpoints.

```python
from trainer_tools.hooks import CheckpointHook

ckpt = CheckpointHook(
    save_dir="checkpoints/",
    save_every_steps=500,
    keep_last=3,
    save_strategy="best",   # or "latest"
    metric_name="valid_loss",
    resume_path="checkpoints/step_1000",  # optional
)
```

#### `EMAHook`
Maintains an exponential moving average (EMA) of model weights. Validation is automatically run against the EMA model, which often gives better generalization. The EMA state is saved and restored with `CheckpointHook`.

```python
from trainer_tools.hooks import EMAHook
ema = EMAHook(decay=0.9999)
```

#### `BatchTransformHook`
Applies on-GPU data augmentations or pre-processing transforms to inputs and/or targets at the start of each batch. Separate transforms can be provided for training and validation.

```python
from trainer_tools.hooks import BatchTransformHook
import torchvision.transforms.v2 as T

aug = BatchTransformHook(
    x_tfm=T.RandomHorizontalFlip(),
    x_tfms_valid=None,   # no aug during validation
)
```

#### `AccelerateHook`
Integrates [HuggingFace Accelerate](https://github.com/huggingface/accelerate) for distributed training (DDP/FSDP), mixed precision, and gradient accumulation in a single hook. When used, do **not** add `AMPHook`, `GradClipHook`, or `GradientAccumulationHook` — Accelerate handles all of that.

```python
from trainer_tools.hooks.accelerate import AccelerateHook

accel = AccelerateHook(
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    mixed_precision="bf16",
)
```