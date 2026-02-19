import logging
from ..imports import *
from ..trainer import Trainer
from .base import BaseHook
from copy import deepcopy

log = logging.getLogger(__name__)


class EMAHook(BaseHook):
    """Keeps Exponential moving average of a model"""

    ord = 20

    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.ema_model = None

    @staticmethod
    def _unwrap(trainer: Trainer):
        """Return the raw model, stripping any DDP/FSDP wrapper."""
        if trainer.is_distributed:
            return trainer.accelerator.unwrap_model(trainer.model)
        return trainer.model

    def before_fit(self, trainer: Trainer):
        self.ema_model = deepcopy(self._unwrap(trainer))
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        if hasattr(trainer, "_ema_state_buffer"):
            log.info("Loading EMA state from checkpoint buffer...")
            self.ema_model.load_state_dict(trainer._ema_state_buffer)
            del trainer._ema_state_buffer

    def after_step(self, trainer: Trainer):
        if not trainer.training:
            return
        model = self._unwrap(trainer)
        with t.no_grad():
            for p_ema, p_model in zip(self.ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(self.decay).add_(p_model.data, alpha=1 - self.decay)

    def before_valid(self, trainer: Trainer):
        self.temp_model = trainer.model
        trainer.model = self.ema_model

    def after_epoch(self, trainer: Trainer):
        if hasattr(self, "temp_model"):
            trainer.model = self.temp_model
            del self.temp_model
