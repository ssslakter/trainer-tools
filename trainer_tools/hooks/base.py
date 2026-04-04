class BaseHook:
    """Base class for hooks. Hooks can interact with the Trainer at various points."""

    ord: int = 0  # Order of execution, higher values run later

    def store(self, trainer, key, value):
        """Stores a value in the trainer.state dictionary under the hook's namespace."""
        trainer.state[f"{self.__class__.__name__}.{key}"] = value

    def get(self, trainer, hook_cls, key, default=None):
        """Retrieves a value from the trainer.state dictionary under another hook's namespace."""
        cls_name = hook_cls.__name__ if isinstance(hook_cls, type) else hook_cls
        return trainer.state.get(f"{cls_name}.{key}", default)

    def before_fit(self, trainer):
        """Called before training starts.
        Guaranteed attributes: trainer.model, trainer.train_dl, trainer.valid_dl, trainer.opt, trainer.train_step, trainer.eval_step, trainer.epochs, trainer.device, trainer.config, trainer.accelerator, trainer.step_state, trainer.state, trainer.result
        """
        pass

    def before_epoch(self, trainer):
        """Called before each epoch (train + val).
        Guaranteed attributes (in addition to above): trainer.start_epoch, trainer.training, trainer.dl
        """
        pass

    def before_step(self, trainer):
        """Called before processing a batch.
        Guaranteed attributes: trainer.batch
        """
        pass

    def after_pred(self, trainer):
        """Called after forward pass. Note: no longer called natively since predict is removed, keep for legacy or user hooks."""
        pass

    def after_loss(self, trainer):
        """Called after loss calculation. Note: no longer called natively since get_loss is removed, keep for legacy or user hooks."""
        pass

    def after_backward(self, trainer):
        """Called after loss.backward().
        Guaranteed attributes (in addition to before_step): trainer.result (has 'loss')
        """
        pass

    def after_step(self, trainer):
        """Called after opt.step() and opt.zero_grad() but before batch logic cleanup.
        Guaranteed attributes: trainer._did_opt_step (True if optimizer stepped)
        """
        pass

    def before_valid(self, trainer):
        """Will be called between train and val dataloaders within an epoch.
        Guaranteed attributes: trainer.training is False, trainer.dl is valid_dl
        """
        pass

    def after_epoch(self, trainer):
        """Called after an entire epoch is finished (train + valid)."""
        pass

    def after_fit(self, trainer):
        """Called fully after the fit block finishes."""
        pass

    def after_cancel(self, trainer):
        """Called when training is interrupted (e.g. KeyboardInterrupt)."""
        pass


class MainProcessHook(BaseHook):
    """
    Marker base class for hooks that should only run on the main process.

    Hooks that inherit from this class will be automatically skipped on
    non-main processes in distributed training, eliminating the need for
    'if trainer.is_main' guards inside the hook implementation.

    Typical use cases:
    - Metrics logging
    - Checkpointing
    - Progress bars
    - Any I/O or console output
    """

    pass


class LambdaHook(BaseHook):
    """Creates a hook from callables passed as keyword arguments."""

    def __init__(self, **callbacks):
        for k, v in callbacks.items():
            setattr(self, k, v)
