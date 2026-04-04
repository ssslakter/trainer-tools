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
    output: dict = field(default_factory=dict)

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
    model or task, provide a `train_step` callback that processes a batch and returns
    a dictionary containing at minimum a 'loss' key.

    You can also add call other functions or add/modify attributes during training by using callbacks. You have to create
    class with methods corresponding to training stages. The available hook points
    are: `begin_fit`, `after_fit`, `begin_epoch`, `after_epoch`, `before_step`,
    `after_step`, `after_backward`. Pass a list
    of hook instances or single hook to the `hooks` parameter during initialization.
    """

    model: nn.Module

    def __init__(
        self,
        model,
        train_step: Callable,
        train_dl=None,
        valid_dl=None,
        eval_step: Callable | None = None,
        optim: t.optim.Optimizer | None = None,
        epochs=10,
        hooks=None,
        device=default_device,
        config=None,
    ):
        self.model, self.train_step, self.train_dl, self.valid_dl, self.opt = model, train_step, train_dl, valid_dl, optim
        self.eval_step = eval_step or train_step
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

    def do_backward(self):
        """Performs backward pass. Can be replaced by hooks or subclasses."""
        if "loss" in self.state.output:
            self.state.output["loss"].backward()
    
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

        step_func = self.train_step if self.model.training else self.eval_step
        output = step_func(self.batch, self)
        
        if not isinstance(output, dict):
            raise TypeError(f"The step function must return a dictionary, but got {type(output).__name__}.")
        self.state.output = output

        if "loss" in output:
            self.loss = output["loss"].item()
            self.loss_t = output["loss"]
        else:
            self.loss = None
            self.loss_t = None
        
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

        self.batch = None
        self.state.output = {}
    
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
        
    def describe_hooks(self):
        """Prints a Markdown table of hooks to tell you what runs when and in what order."""
        from .hooks.base import BaseHook # local import
        points = [
            "before_fit", "before_epoch", "before_step",
            "after_pred", "after_loss", "after_backward",
            "after_step", "before_valid", "after_epoch",
            "after_fit", "after_cancel"
        ]
        
        sorted_hooks = sorted(self.hooks, key=lambda h: getattr(h, "ord", 0))
        lines = ["| Lifecycle Point | Hook | Order |", "| --------------- | ---- | ----- |"]
        
        for point in points:
            active_hooks = []
            for h in sorted_hooks:
                cls_method = getattr(type(h), point, None)
                base_method = getattr(BaseHook, point, None)
                is_dynamic = point in getattr(h, "callbacks", {})
                if cls_method is not base_method or is_dynamic:
                    active_hooks.append(h)
                            
            if not active_hooks:
                lines.append(f"| {point} | (none) | |")
            for i, h in enumerate(active_hooks):
                name = h.__class__.__name__
                order = getattr(h, "ord", 0)
                point_str = point if i == 0 else ""
                lines.append(f"| {point_str} | {name} | {order} |")
        
        print("\n".join(lines))
