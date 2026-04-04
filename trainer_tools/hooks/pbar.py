import numpy as np
from tqdm.auto import tqdm
from .base import MainProcessHook


class ProgressBarHook(MainProcessHook):
    """A hook to display progress bars for epochs and batches."""

    ord = 5

    def __init__(self, freq=10):
        self.freq = freq

    def before_fit(self, trainer):
        self.epoch_bar = tqdm(
            range(trainer.epochs),
            desc="Epoch",
            initial=trainer.step_state.epoch,
            total=trainer.epochs,
        )

    def before_epoch(self, trainer):
        self._init_pbar(
            trainer,
            desc=f"Epoch {trainer.step_state.epoch + 1}/{trainer.epochs} [Train]",
            initial=trainer.step_state.batch_idx,
        )

    def before_valid(self, trainer):
        self.bar.close()
        self._init_pbar(trainer, desc=f"Epoch {trainer.step_state.epoch + 1}/{trainer.epochs} [Valid]")

    def _init_pbar(self, trainer, desc, initial=0):
        total = len(trainer.dl)
        if trainer.is_distributed:
            total *= trainer.accelerator.num_processes
        self.running_loss, self.count = 0.0, 0
        self.bar = tqdm(
            trainer.dl,
            initial=initial,
            total=total,
            desc=desc,
            leave=False,
        )

    def after_step(self, trainer):
        self.running_loss += trainer.loss
        self.count += 1
        self.bar.update(trainer.accelerator.num_processes if trainer.is_distributed else 1)
        if (self.count - 1) % self.freq == 0:
            self.bar.set_postfix(loss=f"{self.running_loss / self.count:.4f}", refresh=False)

    def after_epoch(self, trainer):
        self.epoch_bar.update(1)
        self.bar.close()

    def after_fit(self, _):
        self.epoch_bar.close()
