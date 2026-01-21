from abc import ABC, abstractmethod

from trainer_tools.trainer import Trainer
from ..imports import *

__all__ = ["Metric", "FunctionalMetric", "Loss", "Accuracy"]


class Metric(ABC):
    """
    Base class for data collection strategies.

    Args:
        name: Identifier for the metric.
        freq: How often to collect (in steps).
        phase: The training phase to collect data. Must match a method name
               in `BaseHook` (e.g., 'after_loss', 'after_backward').
    """

    def __init__(self, name: str = None, freq: int = 1, phase: str = "after_loss"):
        self.name = name
        self.freq = freq
        self.phase = phase

    def should_run(self, trainer) -> bool:
        if not trainer.training:
            return True
        return trainer.step % self.freq == 0

    @abstractmethod
    def __call__(self, trainer) -> dict:
        """Return a dictionary of scalar metrics."""
        pass


class FunctionalMetric(Metric):
    """Wrapper to create a metric from a callable."""

    def __init__(self, fn: Callable, name: str = None, freq: int = 1, phase="after_loss"):
        super().__init__(name, freq, phase)
        self.fn = fn

    def __call__(self, trainer):
        return self.fn(trainer)


class Loss(Metric):
    def __init__(self, freq=1):
        super().__init__("loss", freq, phase="after_loss")

    def __call__(self, trainer: Trainer):
        return {self.name: trainer.loss}


class Accuracy(Metric):
    def __init__(self, freq=1):
        super().__init__("accuracy", freq, phase="after_loss")

    def __call__(self, trainer: Trainer):
        if not hasattr(trainer, "preds") or not hasattr(trainer, "yb"):
            return {}
        preds = trainer.preds.argmax(dim=1) if trainer.preds.ndim > 1 else (trainer.preds > 0.5)
        return {self.name: (preds == trainer.yb).float().mean().item()}
