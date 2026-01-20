import logging
from ..trainer import BaseTrainer
from ..imports import *
from torch.amp import GradScaler, autocast
from .base import BaseHook

log = logging.getLogger(__name__)


class LRSchedulerHook(BaseHook):
    """A hook to integrate a PyTorch learning rate scheduler into the training loop."""
    ord = -100  # Run early but after CheckpointHook so it initializes sched before checkpoint loads it

    def __init__(self, sched_fn, step_on_batch=True):
        self.sched_fn, self.step_on_batch = sched_fn, step_on_batch

    def before_fit(self, trainer):
        if isinstance(self.sched_fn, torch.optim.lr_scheduler.LRScheduler):
            self.sched = self.sched_fn
        else:
            self.sched = self.sched_fn(trainer.opt)
        self.lrs = []

    def after_step(self, trainer):
        # We log LR after the step, so it reflects the value used for the update
        self.lrs.append(self.sched.get_last_lr()[0])
        if self.step_on_batch and trainer.training:
            self.sched.step()

    def after_epoch(self, trainer):
        if not self.step_on_batch:
            self.sched.step()

    def plot_lrs(self, ax=None):
        "Plots the learning rate schedule over training steps."
        if not ax:
            _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.lrs)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True)
        plt.show()


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
        if trainer.training:
            trainer.autocast.__enter__()

    def after_loss(self, trainer):
        """
        Called after loss calculation.
        We need to scale the loss *before* the backward pass.
        The base trainer's self.loss_t.backward() will now operate on the scaled loss.
        """
        if not trainer.training:
            return
        trainer.autocast.__exit__(None, None, None)
        trainer.loss_t = trainer.scaler.scale(trainer.loss_t)

    def after_backward(self, trainer):
        """
        Called after backward(). This is where we replace the optimizer step.
        """
        # Unscale the gradients and step the optimizer
        if trainer.loss != 0:
            trainer.scaler.step(trainer.opt)
            # Update the scale for next iteration
            trainer.scaler.update()
        # Signal to the trainer that the step and zero_grad have been handled
        trainer.step_handled_by_hook = True


class EmptyCudaCacheHook(BaseHook):
    def before_valid(self, trainer):
        torch.cuda.empty_cache()


class GradClipHook(BaseHook):
    """Hook to clip gradient"""

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_backward(self, trainer: BaseTrainer):
        nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)
