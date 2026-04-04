import logging
from ..trainer import Trainer
from ..imports import *
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
    except ImportError:
        from torch.cpu.amp import GradScaler
    from torch import autocast
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
        if trainer.training and trainer._did_opt_step:
            self.sched.step()



class AMPHook(BaseHook):
    """
    A hook to seamlessly add Automatic Mixed Precision (AMP).
    - Initializes a GradScaler at the beginning of training.
    - Wraps the forward pass (predict + get_loss) in an autocast context.
    - Wraps backward and optimizer step with gradient scaling.
    """

    def __init__(self, enabled=True, dtype=torch.float16, device_type="cuda"):
        self.enabled, self.dtype, self.device_type = enabled, dtype, device_type

    def before_fit(self, trainer):
        """Called before training starts. Initialize the scaler and wrap operations."""
        has_bf16_params = any(p.dtype == torch.bfloat16 for p in trainer.model.parameters())
        use_scaler = self.enabled and self.dtype == torch.float16 and not has_bf16_params
        trainer.scaler = GradScaler(enabled=use_scaler)
        trainer.autocast = autocast(self.device_type, enabled=self.enabled, dtype=self.dtype)
        log.info(f"Mixed Precision Training: {'Enabled' if self.enabled else 'Disabled'}")
        
        # Wrap operations
        original_backward = trainer.do_backward
        original_opt_step = trainer.do_opt_step
        
        def amp_backward():
            if trainer.loss_t is not None:
                scaled_loss = trainer.scaler.scale(trainer.loss_t)
                scaled_loss.backward()
        
        def amp_opt_step():
            if trainer.loss != 0:
                trainer.scaler.step(trainer.opt)
                trainer.scaler.update()
                return True
            return False
        
        trainer.do_backward = amp_backward
        trainer.do_opt_step = amp_opt_step

    def before_step(self, trainer):
        """Called before the forward pass. Enter the autocast context."""
        trainer.autocast.__enter__()

    def after_loss(self, trainer: Trainer):
        """Called after loss calculation. Exit the autocast context."""
        trainer.autocast.__exit__(None, None, None)


class EmptyCudaCacheHook(BaseHook):
    def before_valid(self, trainer):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GradClipHook(BaseHook):
    """Hook to clip gradients after backward pass."""
    ord = -5  # Run after backward but before opt step

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_backward(self, trainer: Trainer):
        if not trainer.training:
            return
            
        # Unscale gradients if using AMP
        if trainer.get_hook(AMPHook, None) and hasattr(trainer, 'scaler') and trainer.scaler.is_enabled():
            trainer.scaler.unscale_(trainer.opt)
        nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)

class GradientAccumulationHook(BaseHook):
    """
    Accumulates gradients over multiple steps.
    """

    ord = -10

    def __init__(self, steps: int = 1):
        self.steps = steps
    
    def before_fit(self, trainer):
        """Configure StepState and wrap optimizer operations."""
        trainer.state.grad_accum_steps = self.steps
        
        # Wrap operations to skip step/zero_grad when accumulating
        original_opt_step = trainer.do_opt_step
        original_zero_grad = trainer.do_zero_grad
        
        def grad_accum_opt_step():
            is_last_batch = (trainer.state.batch_idx + 1) >= len(trainer.dl)
            if trainer.state.should_step_optimizer(is_last_batch):
                return original_opt_step()
            return False  # Skipped
        
        def grad_accum_zero_grad():
            is_last_batch = (trainer.state.batch_idx + 1) >= len(trainer.dl)
            if trainer.state.should_step_optimizer(is_last_batch):
                original_zero_grad()
        
        trainer.do_opt_step = grad_accum_opt_step
        trainer.do_zero_grad = grad_accum_zero_grad

    def after_loss(self, trainer):
        """Scale loss by accumulation steps."""
        if trainer.training and trainer.loss_t is not None:
            trainer.loss_t = trainer.loss_t / self.steps
