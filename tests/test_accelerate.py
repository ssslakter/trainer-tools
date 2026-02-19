import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader

from trainer_tools.trainer import Trainer
from trainer_tools.hooks.base import BaseHook
from trainer_tools.hooks.accelerate import AccelerateHook
from trainer_tools.hooks import CheckpointHook, LRSchedulerHook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class LinearModel(nn.Module):
    """Single-weight model for deterministic gradient math."""

    def __init__(self, init_weight: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=False)
        self.fc.weight.data.fill_(init_weight)

    def forward(self, x):
        return self.fc(x)


class CancelAtStepHook(BaseHook):
    """Raises KeyboardInterrupt once trainer reaches *step* on *epoch*."""
    ord = 100

    def __init__(self, step: int, epoch: int = 0):
        self._step, self._epoch = step, epoch

    def after_step(self, trainer):
        if trainer.step >= self._step and trainer.epoch >= self._epoch:
            raise KeyboardInterrupt("CancelAtStepHook")


def _make_constant_loader(x_val=1.0, y_val=0.0, n=4, batch_size=1):
    """DataLoader of (x, y) constant pairs."""
    x = torch.full((1, 1), x_val)
    y = torch.full((1, 1), y_val)
    return DataLoader([(x, y)] * n, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_accelerate_basic_training(simple_model, tuple_loaders):
    """AccelerateHook should run a basic training loop without errors."""
    train_dl, valid_dl = tuple_loaders
    model = simple_model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt,
        loss_func=nn.CrossEntropyLoss(),
        epochs=2,
        hooks=[AccelerateHook()],
        device="cpu",
    )
    trainer.fit()

    assert trainer.step > 0
    assert trainer.epoch == 1  # 0-indexed, finished epoch index


def test_accelerate_checkpoint_save_and_resume(simple_model, tuple_loaders, tmp_path):
    """Checkpoint saved under AccelerateHook should restore model weights and step."""
    train_dl, valid_dl = tuple_loaders
    save_dir = tmp_path / "ckpts"

    # --- first run: train for 10 steps then interrupt -----------------------
    model_1 = simple_model
    opt_1 = torch.optim.Adam(model_1.parameters(), lr=1e-3)

    trainer_1 = Trainer(
        model=model_1,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt_1,
        loss_func=nn.CrossEntropyLoss(),
        epochs=4,
        hooks=[
            AccelerateHook(),
            CheckpointHook(save_dir=save_dir, save_every_steps=5, keep_last=5),
            CancelAtStepHook(step=10, epoch=2),
        ],
        device="cpu",
    )
    with pytest.raises(KeyboardInterrupt):
        trainer_1.fit()

    ckpt_path = save_dir / "checkpoint_interrupted.pt"
    assert ckpt_path.exists()

    # Snapshot weights from first run
    weights_before = {
        k: v.clone()
        for k, v in trainer_1.accelerator.unwrap_model(model_1).state_dict().items()
    }

    # --- second run: resume and verify immediately --------------------------
    model_2 = type(simple_model)()
    opt_2 = torch.optim.Adam(model_2.parameters(), lr=1e-3)

    class VerifyResumeHook(BaseHook):
        ord = 200

        def before_fit(self, trainer):
            assert trainer.epoch == 2
            assert trainer.step == 10
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            for k, v in unwrapped.state_dict().items():
                assert torch.equal(v, weights_before[k]), f"Mismatch in {k}"
            raise KeyboardInterrupt("Verification OK")

    trainer_2 = Trainer(
        model=model_2,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt_2,
        loss_func=nn.CrossEntropyLoss(),
        epochs=4,
        hooks=[
            AccelerateHook(),
            CheckpointHook(save_dir=save_dir, resume_path=str(ckpt_path), save_every_steps=5),
            VerifyResumeHook(),
        ],
        device="cpu",
    )
    with pytest.raises(KeyboardInterrupt, match="Verification OK"):
        trainer_2.fit()


def test_accelerate_gradient_accumulation():
    """Gradient accumulation via AccelerateHook should match the standalone hook result.

    Setup (identical to test_grad_accum.py):
        - Model: y = w*x, w₀ = 1.0
        - Data:  x = 1, y = 0   (4 samples, batch_size=1)
        - Loss:  MSE → L = w²,  dL/dw = 2w
        - Accumulation steps = 2 ⇒ loss divided by 2 → effective grad per sample = w

    Expected trajectory:
        micro-batch 1: w=1.0, grad+=1.0           (accumulate)
        micro-batch 2: w=1.0, grad+=1.0 → tot 2.0 (update) → w = 1.0 − 0.1×2.0 = 0.8
        micro-batch 3: w=0.8, grad+=0.8           (accumulate)
        micro-batch 4: w=0.8, grad+=0.8 → tot 1.6 (update) → w = 0.8 − 0.1×1.6 = 0.64
    """
    torch.manual_seed(42)
    model = LinearModel(init_weight=1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dl = _make_constant_loader(x_val=1.0, y_val=0.0, n=4, batch_size=1)

    trainer = Trainer(
        model=model,
        train_dl=dl,
        valid_dl=dl,
        optim=opt,
        loss_func=nn.MSELoss(),
        epochs=1,
        hooks=[AccelerateHook(gradient_accumulation_steps=2)],
        device="cpu",
    )
    trainer.fit()

    final_w = model.fc.weight.item()
    assert abs(final_w - 0.64) < 1e-5, f"Expected 0.64, got {final_w}"


def test_accelerate_grad_accum_with_lr_scheduler():
    """LRSchedulerHook should only step on actual optimizer-update boundaries.

    With accum_steps=2 and 4 micro-batches, there are 2 real optimizer steps.
    A StepLR(step_size=1, gamma=0.5) should halve the LR twice:
        initial lr=0.1 → after update 1: 0.05 → after update 2: 0.025
    """
    torch.manual_seed(0)
    model = LinearModel(init_weight=1.0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dl = _make_constant_loader(n=4, batch_size=1)

    sched_hook = LRSchedulerHook(
        lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=0.5)
    )
    accel_hook = AccelerateHook(gradient_accumulation_steps=2)

    trainer = Trainer(
        model=model,
        train_dl=dl,
        valid_dl=dl,
        optim=opt,
        loss_func=nn.MSELoss(),
        epochs=1,
        hooks=[accel_hook, sched_hook],
        device="cpu",
    )
    trainer.fit()

    # 2 scheduler steps with gamma=0.5: 0.1 → 0.05 → 0.025
    assert abs(sched_hook.lr - 0.025) < 1e-7, f"Expected lr=0.025, got {sched_hook.lr}"
