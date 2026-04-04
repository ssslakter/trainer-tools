from abc import ABC, abstractmethod

from trainer_tools.trainer import Trainer
from ...imports import *

__all__ = ["Metric", "Loss", "Accuracy"]


class Metric(ABC):
    """
    Base class for data collection strategies.

    Args:
        name: Identifier for the metric.
        freq: How often to collect (in steps) during TRAINING.
              Validation always collects every step.
        phase: The hook method name where collection occurs (e.g. 'after_loss').
        use_prefix: If True (default), keys are prefixed with 'train_'/'valid_'.
                    If False, keys are logged exactly as returned (e.g. 'grad_norm').
    """

    def __init__(self, name: str = None, freq: int = 1, phase: str = "after_step", use_prefix: bool = True):
        self.name = name
        self.freq = freq
        self.phase = phase
        self.use_prefix = use_prefix

    def should_run(self, trainer) -> bool:
        if not trainer.training:
            return True
        return trainer.state.optimizer_step % self.freq == 0

    def get_value(self, trainer, key, fn=None):
        """Helper to extract a value from state.output or compute it via fn."""
        if fn is not None:
            return fn(trainer.state)
        if key not in trainer.state.output:
            cb = "train_step" if trainer.model.training else "eval_step"
            raise KeyError(f"Metric requested key '{key}' but it was not returned by {cb}.")
        return trainer.state.output[key]

    @abstractmethod
    def __call__(self, trainer) -> dict:
        """Return a dictionary of scalar metrics."""
        pass


class Loss(Metric):
    def __init__(self, freq=1, loss_key="loss", loss_fn=None):
        super().__init__("loss", freq, phase="after_step")
        self.loss_key = loss_key
        self.loss_fn = loss_fn

    def __call__(self, trainer: Trainer):
        val = self.get_value(trainer, self.loss_key, self.loss_fn)

        if isinstance(val, torch.Tensor):
            val = val.item()

        return {self.name: val}


class Accuracy(Metric):
    def __init__(self, name="accuracy", freq=1, pred_key="preds", target_key="targets", pred_fn=None, target_fn=None):
        super().__init__(name, freq, phase="after_step")
        self.pred_key = pred_key
        self.target_key = target_key
        self.pred_fn = pred_fn
        self.target_fn = target_fn

    def __call__(self, trainer: Trainer):
        if self.pred_fn is not None:
            preds = self.pred_fn(trainer.state)
        else:
            logits = self.get_value(trainer, self.pred_key)
            preds = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0.5)

        target = self.get_value(trainer, self.target_key, self.target_fn)

        return {self.name: (preds == target).float().mean().item()}
