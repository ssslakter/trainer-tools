from typing import Callable, Optional
from .base import BaseHook
from ..trainer import Trainer
from ..utils import to_device


class BatchTransformHook(BaseHook):
    """Applies batch transform to the input data."""

    def __init__(self, transform: Optional[Callable] = None):
        """
        Args:
            transform (Callable or None): A function that applies transform to a batch.
        """
        self.transform = transform

    def before_step(self, trainer: Trainer):
        if self.transform is not None:
            batch = to_device(trainer.batch, trainer.device)
            trainer.batch = self.transform(*batch)
