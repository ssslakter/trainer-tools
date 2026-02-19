import numpy as np
from tqdm.auto import tqdm
from .base import BaseHook


class ProgressBarHook(BaseHook):
    """A hook to display progress bars for epochs and batches."""

    ord = 5

    def __init__(self, freq=10):
        self.freq = freq

    def before_fit(self, trainer):
        self.step = getattr(trainer, "step", 0)
        self.epoch_bar = tqdm(
            range(trainer.epochs),
            desc="Epoch",
            initial=getattr(trainer, "epoch", 0),
            total=trainer.epochs,
            disable=not trainer.is_main,
        )

    def before_epoch(self, trainer):
        self._init_pbar(
            trainer,
            desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Train]",
            initial=trainer.step % len(trainer.train_dl),
        )

    def before_valid(self, trainer):
        self._init_pbar(trainer, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Valid]")

    def _init_pbar(self, trainer, desc, initial=0):
        self.running_loss, self.count = 0.0, 0
        trainer.dl = self.bar = tqdm(
            trainer.dl,
            initial=initial,
            desc=desc,
            leave=False,
            disable=not trainer.is_main,
        )

    def after_step(self, trainer):
        if not trainer.is_main:
            return
        self.running_loss += trainer.loss
        self.count += 1
        if (self.count - 1) % self.freq == 0:
            self.bar.set_postfix(loss=f"{self.running_loss / self.count:.4f}", refresh=False)

    def after_epoch(self, trainer):
        self.epoch_bar.update(1)
        self.bar.close()

    def after_fit(self, _):
        self.epoch_bar.close()
