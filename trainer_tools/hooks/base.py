class BaseHook:
    """Base class for hooks. Hooks can interact with the Trainer at various points."""
    
    ord: int = 0  # Order of execution, higher values run later

    def before_fit(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def before_step(self, trainer):
        pass

    def after_pred(self, trainer):
        pass

    def after_loss(self, trainer):
        pass

    def after_backward(self, trainer):
        pass

    def after_step(self, trainer):
        pass

    def before_valid(self, trainer):
        """Will be called between train and val dataloaders within an epoch"""
        pass

    def after_epoch(self, trainer):
        pass

    def after_fit(self, trainer):
        pass
    
    def after_cancel(self, trainer):
        """Called when training is interrupted (e.g. KeyboardInterrupt)"""
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