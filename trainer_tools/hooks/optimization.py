import logging
from ..trainer import Trainer
from ..imports import *
from torch.amp import GradScaler, autocast
from .base import BaseHook

log = logging.getLogger(__name__)


class LRSchedulerHook(BaseHook):
    """A hook to integrate a PyTorch learning rate scheduler into the training loop."""

    ord = -100  # Run early but after CheckpointHook so it initializes sched before checkpoint loads it

    def __init__(self, sched_fn):
        self.sched_fn = sched_fn

    @property
    def lr(self):
        return self.sched.get_last_lr()[0]

    def before_fit(self, trainer):
        if isinstance(self.sched_fn, torch.optim.lr_scheduler.LRScheduler):
            self.sched = self.sched_fn
        else:
            self.sched = self.sched_fn(trainer.opt)

    def after_step(self, trainer):
        if trainer.training and not getattr(trainer, "skip_zero_grad", False):
            self.sched.step()



class AMPHook(BaseHook):
    """
    A hook to seamlessly add Automatic Mixed Precision (AMP).
    - Initializes a GradScaler at the beginning of training.
    - Wraps the forward pass (predict + get_loss) in an autocast context.
    - Replaces the standard backward pass with scaler.scale(loss).backward().
    - Replaces the standard optimizer step with scaler.step(optimizer).
    """

    def __init__(self, enabled=True, dtype=torch.float16, device_type="cuda"):
        self.enabled, self.dtype, self.device_type = enabled, dtype, device_type

    def before_fit(self, trainer):
        """Called before training starts. Initialize the scaler."""
        trainer.scaler = GradScaler(enabled=self.enabled)
        # We manually control the autocast context manager
        trainer.autocast = autocast(self.device_type, enabled=self.enabled, dtype=self.dtype)
        log.info(f"Mixed Precision Training: {'Enabled' if self.enabled else 'Disabled'}")

    def before_step(self, trainer):
        """Called before the forward pass. Enter the autocast context."""
        trainer.autocast.__enter__()

    def after_loss(self, trainer: Trainer):
        """
        Called after loss calculation.
        We need to scale the loss *before* the backward pass.
        The base trainer's self.loss_t.backward() will now operate on the scaled loss.
        """
        trainer.autocast.__exit__(None, None, None)
        if trainer.training:
            trainer.loss_t = trainer.scaler.scale(trainer.loss_t)

    def after_backward(self, trainer):
        """
        Called after backward(). This is where we replace the optimizer step.
        """
        if getattr(trainer, "skip_opt_step", False):
            return

        if trainer.loss != 0:
            trainer.scaler.step(trainer.opt)
            trainer.scaler.update()
        trainer.skip_opt_step = True


class EmptyCudaCacheHook(BaseHook):
    def before_valid(self, trainer):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GradClipHook(BaseHook):
    """Hook to clip gradient"""
    ord = AMPHook.ord - 1

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_backward(self, trainer: Trainer):
        if getattr(trainer, "skip_zero_grad", False):
            return

        if trainer.get_hook(AMPHook, None):
            trainer.scaler.unscale_(trainer.opt)
        nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)

class GradientAccumulationHook(BaseHook):
    """
    Accumulates gradients over multiple steps.
    """

    ord = -10

    def __init__(self, steps: int = 1):
        self.steps = steps

    def after_loss(self, trainer):
        if trainer.training:
            trainer.loss_t = trainer.loss_t / self.steps

    def after_backward(self, trainer):
        if not trainer.training:
            return

        step_idx = trainer.step + 1
        is_update = (step_idx % self.steps == 0)

        # Also check end of epoch
        if hasattr(trainer, "dl") and hasattr(trainer, "batch_idx"):
            try:
                if trainer.batch_idx + 1 == len(trainer.dl):
                    is_update = True
            except TypeError:
                pass  # len() might fail

        if not is_update:
            trainer.skip_opt_step = True
            trainer.skip_zero_grad = True
