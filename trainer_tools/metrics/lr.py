from .base import Metric
from ..hooks import LRSchedulerHook
from ..trainer import Trainer
from ..imports import *


class LRStats(Metric):
    """Collects learning rate statistics."""

    def __init__(self, name="lr", freq=1):
        super().__init__(name, freq, phase="after_step", use_prefix=False)

    def should_run(self, trainer):
        if not trainer.training:
            return False
        return super().should_run(trainer)

    def __call__(self, trainer: Trainer):
        return {self.name: trainer.get_hook(LRSchedulerHook).lrs[-1]}
