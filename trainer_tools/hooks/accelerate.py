import logging
from accelerate import Accelerator
from ..trainer import Trainer
from .base import BaseHook
from .optimization import AMPHook, GradientAccumulationHook, GradClipHook, LRSchedulerHook

log = logging.getLogger(__name__)


class AccelerateHook(BaseHook):
    """
    Integrates HF Accelerate into the training loop.

    Handles distributed training (DDP/FSDP), mixed precision, gradient accumulation,
    gradient clipping, and device placement — all through a single hook.

    When using AccelerateHook, do **not** add ``AMPHook``, ``GradientAccumulationHook``,
    or ``GradClipHook`` — their functionality is subsumed by Accelerate.
    ``LRSchedulerHook`` remains compatible and its scheduler will be prepared automatically.

    Args:
        gradient_accumulation_steps: Number of micro-batches to accumulate before
            an optimizer update. Accelerate handles loss scaling and gradient
            synchronisation suppression automatically.
        max_grad_norm: Maximum gradient norm for clipping (applied only on
            synchronisation/update steps). ``None`` disables clipping.
        **kwargs: Forwarded to ``accelerate.Accelerator()``.
            Useful options include ``mixed_precision`` (``"fp16"``, ``"bf16"``, ``"no"``),
            ``gradient_accumulation_plugin``, ``log_with``, ``project_dir``, etc.
    """

    ord = -50

    def __init__(
        self,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        **kwargs,
    ):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.max_grad_norm = max_grad_norm
        self._accumulate_ctx = None

    def before_fit(self, trainer: Trainer):
        incompatible = {AMPHook, GradientAccumulationHook, GradClipHook}
        found = [type(h).__name__ for h in trainer.hooks if type(h) in incompatible]
        assert not found, (
            f"AccelerateHook is not compatible with {', '.join(found)}. "
            "Remove them and configure via AccelerateHook / Accelerator kwargs instead."
        )

        trainer.accelerator = self.accelerator
        trainer.device = self.accelerator.device

        trainer.model, trainer.opt, trainer.train_dl, trainer.valid_dl = (
            self.accelerator.prepare(
                trainer.model, trainer.opt, trainer.train_dl, trainer.valid_dl
            )
        )

        if hook := trainer.get_hook(LRSchedulerHook, None):
            hook.sched = self.accelerator.prepare(hook.sched)

        log.info(
            "AccelerateHook initialised — device: %s, mixed-precision: %s, "
            "grad-accum steps: %s, distributed: %s",
            self.accelerator.device,
            self.accelerator.mixed_precision,
            self.accelerator.gradient_accumulation_steps,
            self.accelerator.distributed_type,
        )


    def before_step(self, trainer: Trainer):
        if trainer.training:
            self._accumulate_ctx = self.accelerator.accumulate(trainer.model)
            self._accumulate_ctx.__enter__()

    def after_loss(self, trainer: Trainer):
        if trainer.training:
            self.accelerator.backward(trainer.loss_t)
            trainer.skip_backward = True

    def after_backward(self, trainer: Trainer):
        if self.max_grad_norm and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                trainer.model.parameters(), self.max_grad_norm
            )

    def after_step(self, trainer: Trainer):
        if trainer.training and self._accumulate_ctx is not None:
            self._accumulate_ctx.__exit__(None, None, None)
            self._accumulate_ctx = None