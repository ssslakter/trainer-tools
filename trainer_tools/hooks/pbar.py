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
            range(trainer.epochs), desc="Epoch", initial=getattr(trainer, "epoch", 0), total=trainer.epochs
        )

    def before_epoch(self, trainer):
        self.running_loss, self.count = 0.0, 0
        trainer.dl = self.bar = tqdm(
            trainer.dl,
            initial=trainer.step % len(trainer.dl),
            desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Train]",
            leave=False,
        )

    def before_valid(self, trainer):
        self.running_loss, self.count = 0.0, 0
        trainer.dl = self.bar = tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Valid]", leave=False)

    def after_step(self, trainer):
        self.running_loss += trainer.loss
        self.count += 1
        if self.count % self.freq == 0:
            self.bar.set_postfix(loss=f"{self.running_loss / self.count:.4f}", refresh=False)

    def after_epoch(self, trainer):
        self.epoch_bar.update(1)
        self.bar.close()

    def after_fit(self, _):
        self.epoch_bar.close()
