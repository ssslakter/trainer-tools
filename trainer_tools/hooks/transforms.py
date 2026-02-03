from typing import Callable, Union
from .base import BaseHook
from ..trainer import Trainer
from ..utils import to_device


class BatchTransformHook(BaseHook):
    """Applies batch transform to the input data."""

    def __init__(self, transforms: Union[list, Callable, None] = None):
        """
        Args:
            transforms (Callable or None): A function that applies transform to a batch.
        """
        self.transforms = transforms
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        elif self.transforms is None:
            self.transforms = []

    def before_step(self, trainer: Trainer):
        for tfm in self.transforms:
            batch = to_device(trainer.batch, trainer.device)
            trainer.batch = tfm(*batch)
