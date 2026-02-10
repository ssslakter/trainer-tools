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

## The Hook System

`trainer-tools` relies on `BaseHook`. You can create custom behavior by subclassing it:

```python
from trainer_tools.hooks import BaseHook

class MyCustomHook(BaseHook):
    def after_step(self, trainer):
        if trainer.step % 100 == 0:
            print(f"Current Loss: {trainer.loss}")
```