import numpy as np
from tqdm.auto import tqdm
from .base import BaseHook


class ProgressBarHook(BaseHook):
    """A hook to display progress bars for epochs and batches."""

    ord = 5

    def before_fit(self, trainer):
        self.step = getattr(trainer, "step", 0)
        self.epoch_bar = tqdm(
            range(trainer.epochs), desc="Epoch", initial=getattr(trainer, "epoch", 0), total=trainer.epochs
        )

    def before_epoch(self, trainer):
        self.losses = []
        trainer.dl = self.bar = tqdm(
            trainer.dl, initial=self.step, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Train]", leave=False
        )

    def before_valid(self, trainer):
        self.losses = []
        trainer.dl = self.bar = tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Valid]", leave=False)

    def after_step(self, trainer):
        self.losses.append(trainer.loss)
        self.bar.set_postfix(loss=f"{np.mean(self.losses):.4f}")

    def after_epoch(self, trainer):
        self.epoch_bar.update(1)

    def after_fit(self, _):
        self.epoch_bar.close()
