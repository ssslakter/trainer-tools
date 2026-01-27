from typing import Type, TypeVar

from .imports import *
from .utils import default_device, to_device

T = TypeVar("T")
RAISE = object()


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

    model: nn.Module

    def __init__(
        self,
        model,
        train_dl=None,
        valid_dl=None,
        optim: t.optim.Optimizer = None,
        loss_func: Callable = None,
        epochs=10,
        hooks=None,
        device=default_device,
        config=None,
    ):
        self.model, self.train_dl, self.valid_dl, self.opt, self.loss_func = model, train_dl, valid_dl, optim, loss_func
        self.epochs, self.hooks = epochs, hooks if hooks else []
        self.device, self.config = device, config

    def _call_hook(self, method_name):
        sorted_hooks = sorted(self.hooks, key=lambda h: getattr(h, "ord", 0))
        for hook in sorted_hooks:
            getattr(hook, method_name, lambda trainer: None)(self)

    def get_loss(self) -> torch.Tensor:
        """Calculates the loss for the current batch. Can be overwritten by a subclass."""
        if isinstance(self.preds, dict) and "loss" in self.preds:
            return self.preds["loss"]

        if self.loss_func is None:
            return None
        return self.loss_func(self.preds, self.yb)

    def predict(self, xb) -> dict:
        """Performs a forward pass on the model. Can be overwritten by a subclass."""
        return self.model(xb)

    def _one_batch(self):
        """Process single batch forward, optionally with backward"""
        self._call_hook("before_step")

        self.xb, self.yb = to_device(self.batch, self.device)
        self.step_handled_by_hook = False
        self.preds = self.predict(self.xb)
        
        self._call_hook("after_pred")
        self.loss_t = self.get_loss()
        if self.loss_t is not None:
            self.loss = self.loss_t.item()
        self._call_hook("after_loss")
        if self.model.training:
            self.loss_t.backward()
            self._call_hook("after_backward")
            if not self.step_handled_by_hook:
                self.opt.step()
                self.opt.zero_grad()
            else:
                self.opt.zero_grad()
            self.step += 1
        self._call_hook("after_step")

    def _one_epoch(self):
        """Run single epoch"""
        for self.batch_idx, self.batch in enumerate(self.dl):
            self._one_batch()

    def evaluate(self, valid_dl=None):
        """Evaluates the model on the validation dataset."""
        self.epoch = self.step = 0
        self.n_steps = len(self.train_dl) * self.epochs
        self.model.to(self.device)
        self._call_hook("before_fit")
        self.model.eval()
        self.training = False
        self.dl = valid_dl if valid_dl is not None else self.valid_dl
        self._call_hook("before_valid")
        with torch.no_grad():
            self._one_epoch()
        self._call_hook("after_epoch")

    def fit(self):
        """Starts the training and validation loops for the specified number of epochs."""
        self.n_steps = len(self.train_dl) * self.epochs
        self.step = 0
        self.model.to(self.device)
        self._call_hook("before_fit")
        try:
            for self.epoch in range(self.epochs):
                # Train
                self.model.train()
                self.training, self.dl = True, self.train_dl
                self._call_hook("before_epoch")
                self._one_epoch()

                # Validation
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
