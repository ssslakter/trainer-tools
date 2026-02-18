import accelerate
from accelerate import Accelerator
from ..trainer import Trainer
from . import BaseHook, AMPHook, GradientAccumulationHook, LRSchedulerHook


class AccelerateHook(BaseHook):
    """
    Integrates HF Accelerate.
    Handles DDP, mixed precision, and device placement.
    """

    ord = -50

    def __init__(self, **kwargs):
        self.accelerator = Accelerator(**kwargs)

    def before_fit(self, trainer: Trainer):
        assert (
            trainer.get_hook(AMPHook, None) is None and trainer.get_hook(GradientAccumulationHook, None) is None
        ), "AccelerateHook is not compatible with AMP or GradientAccumulation hooks. Please remove them when using AccelerateHook."
        trainer.accelerator = self.accelerator
        trainer.device = self.accelerator.device

        trainer.model, trainer.opt, trainer.train_dl, trainer.valid_dl = self.accelerator.prepare(
            trainer.model, trainer.opt, trainer.train_dl, trainer.valid_dl
        )
        if hook:=trainer.get_hook(LRSchedulerHook, None):
            hook.sched = trainer.accelerator.prepare(hook.sched)
        