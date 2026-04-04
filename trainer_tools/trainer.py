from typing import Type, TypeVar
from dataclasses import dataclass, field
from accelerate import Accelerator
from .imports import *
from .utils import default_device, to_device

T = TypeVar("T")
RAISE = object()


@dataclass
class StepState:
    """Centralized step/batch/sample counting, invariant to num_processes and grad accumulation."""
    batch_idx: int = 0        # within current epoch
    optimizer_step: int = 0   # increments after opt.step()
    samples_seen: int = 0     # global across all processes, use as log x-axis
    epoch: int = 0
    grad_accum_steps: int = 1
    num_processes: int = 1

    def should_step_optimizer(self, is_last_batch: bool = False) -> bool:
        return (self.batch_idx + 1) % self.grad_accum_steps == 0 or is_last_batch

    def increment_batch(self, batch_size: int, is_training: bool = True, did_optimizer_step: bool = False):
        self.batch_idx += 1
        if is_training:
            self.samples_seen += batch_size * self.num_processes
            if did_optimizer_step:
                self.optimizer_step += 1

    def reset_epoch(self):
        self.batch_idx = 0

class Trainer:
    """
    Helper class that contains training and evaluation loop.

    This class provides a generic framework for training. To adapt it for a specific
    model or task, you can extend its functionality in two primary ways:

    1. Inheritance:
       Create a new class that inherits from `BaseTrainer`. You MUST implement
       the following methods:
       - `predict(self, xb)`: Defines how the model processes an input batch `xb`.
         The results should be stored as instance attributes (e.g., `self.preds`)
         to be used by the loss function.
       - `get_loss(self)`: Defines how the loss is calculated using the outputs
         from `predict()` and the ground truth `self.yb`.

    2. Hooks (Callbacks):
        You can also add call other functions or add/modify attributes during training by using callbacks. You have to create
       class with methods corresponding to training stages. The available hook points
       are: `begin_fit`, `after_fit`, `begin_epoch`, `after_epoch`, `begin_step`,
       `after_step`, `after_pred`, `after_loss`, and `after_backward`. Pass a list
       of hook instances or single hook to the `hooks` parameter during initialization.
    """

    target_keys: list = ["y", "targets", "labels", "target", "label"]
    model: nn.Module

    def __init__(
        self,
        model,
        train_dl=None,
        valid_dl=None,
        optim: t.optim.Optimizer | None = None,
        loss_func: Callable | None = None,
        epochs=10,
        hooks=None,
        device=default_device,
        config=None,
    ):
        self.model, self.train_dl, self.valid_dl, self.opt, self.loss_func = model, train_dl, valid_dl, optim, loss_func
        self.epochs, self.hooks = epochs, hooks if hooks else []
        self.device, self.config = device, config
        self.accelerator: Accelerator = None
        self.state = StepState()

    @property
    def is_main(self):
        return not self.is_distributed or self.accelerator.is_main_process

    @property
    def is_distributed(self):
        return self.accelerator is not None

    def _call_hook(self, method_name):
        from .hooks.base import MainProcessHook
        sorted_hooks = sorted(self.hooks, key=lambda h: getattr(h, "ord", 0))
        for hook in sorted_hooks:
            # Skip MainProcessHook instances on non-main processes
            if isinstance(hook, MainProcessHook) and not self.is_main:
                continue
            getattr(hook, method_name, lambda trainer: None)(self)

    def get_loss(self, preds, target) -> torch.Tensor:
        """Calculates the loss for the current batch. Can be overwritten by a subclass."""
        if isinstance(preds, dict) and "loss" in preds:
            return preds["loss"]

        if self.loss_func is None:
            raise ValueError("No loss function provided. Please implement get_loss, provide loss_func or return loss from the model")
        return self.loss_func(preds, target)

    def get_input(self, batch):
        """Extracts inputs from the batch. Can be overwritten by a subclass."""
        if isinstance(batch, (list, tuple)):
            xb, _ = batch
            return xb
        elif isinstance(batch, dict):
            return {k: v for k, v in batch.items() if k not in self.target_keys}
        return batch

    def get_target(self, batch):
        """Extracts targets from the batch. Can be overwritten by a subclass."""
        if isinstance(batch, (list, tuple)):
            _, yb = batch
            return yb
        elif isinstance(batch, dict):
            for key in self.target_keys:
                if key in batch:
                    return batch[key]
        # if no target key is found, return input in case of self-supervised learning
        return self.get_input(batch)

    def predict(self, batch) -> dict:
        """Performs a forward pass on the model. Can be overwritten by a subclass."""
        inputs = self.get_input(batch)
        if isinstance(inputs, dict):
            return self.model(**inputs)
        return self.model(inputs)
    
    def do_backward(self):
        """Performs backward pass. Can be replaced by hooks or subclasses."""
        if self.loss_t is not None:
            self.loss_t.backward()
    
    def do_opt_step(self) -> bool:
        """
        Performs optimizer step. Can be replaced by hooks or subclasses.
        Returns True if optimizer step was actually performed, False if skipped.
        """
        self.opt.step()
        return True
    
    def do_zero_grad(self):
        """Zeros gradients. Can be replaced by hooks or subclasses."""
        self.opt.zero_grad()

    def _one_batch(self):
        """Process single batch forward, optionally with backward"""
        self._call_hook("before_step")
        if not self.is_distributed:
            self.batch = to_device(self.batch, self.device)

        self.preds = self.predict(self.batch)

        self._call_hook("after_pred")
        self.loss_t = self.get_loss(self.preds, self.get_target(self.batch))
        if self.loss_t is not None:
            self.loss = self.loss_t.item()
        self._call_hook("after_loss")
        
        if self.model.training:
            self.do_backward()
            self._call_hook("after_backward")
            
            # Step optimizer and track whether it actually stepped
            self._did_opt_step = self.do_opt_step()
            self.do_zero_grad()
        else:
            self._did_opt_step = False
            
        self._call_hook("after_step")
        
        # Update state after the step
        if self.model.training:
            batch_size = self._get_batch_size(self.batch)
            self.state.increment_batch(batch_size, is_training=True, did_optimizer_step=self._did_opt_step)

        self.loss_t = self.preds = None
    
    def _get_batch_size(self, batch) -> int:
        """Extract batch size from batch for state tracking."""
        if isinstance(batch, (list, tuple)):
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        elif isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        return 1  # fallback

    def _one_epoch(self):
        """Run single epoch"""
        self.state.reset_epoch()
        for _, self.batch in enumerate(self.dl):
            self._one_batch()
        self.batch = None

    def fit(self):
        """Starts the training and validation loops for the specified number of epochs."""
        # Initialize state
        self.state.num_processes = self.accelerator.num_processes if self.is_distributed else 1
        self.start_epoch = 0
        
        self._call_hook("before_fit")
        self.model.to(self.device)
        try:
            for epoch_idx in range(self.start_epoch, self.epochs):
                self.state.epoch = epoch_idx
                
                # Train
                self.model.train()
                self.training, self.dl = True, self.train_dl
                self._call_hook("before_epoch")
                self._one_epoch()

                # Validation
                if self.valid_dl is not None:
                    self.model.eval()
                    self.training, self.dl = False, self.valid_dl
                    self._call_hook("before_valid")
                    with torch.no_grad():
                        self._one_epoch()
                self._call_hook("after_epoch")
        except KeyboardInterrupt:
            self._call_hook("after_cancel")
            raise
        self._call_hook("after_fit")

    def get_hook(self, cls: Type[T], default=RAISE) -> T:
        for h in self.hooks:
            if isinstance(h, cls):
                return h
        if default is not RAISE:
            return default
        raise KeyError(f"Hook {cls} not found")
