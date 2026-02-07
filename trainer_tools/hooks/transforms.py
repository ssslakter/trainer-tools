from typing import Callable, Union, Optional
import torch
from .base import BaseHook
from ..trainer import Trainer
from ..utils import to_device


class BatchTransformHook(BaseHook):
    """Applies batch transform to the input data."""

    def __init__(
        self,
        x_tfm: Optional[Callable] = None,
        y_tfm: Optional[Callable] = None,
        batch_tfms: Union[list, Callable, None] = None,
        x_tfms_valid: Optional[Callable] = None,
        y_tfms_valid: Optional[Callable] = None,
    ):
        """
        Args:
            x_tfm (Callable, optional): Transformation to apply to inputs. Defaults to None.
            y_tfm (Callable, optional): Transformation to apply to targets. Defaults to None.
            batch_tfms (list or Callable, optional): List of batch transformations to apply. Defaults to None.
        """
        self.x_tfm, self.y_tfm = x_tfm, y_tfm
        self.x_tfms_valid, self.y_tfms_valid = x_tfms_valid, y_tfms_valid
        self.batch_tfms = batch_tfms or []
        if not isinstance(self.batch_tfms, list):
            self.batch_tfms = [self.batch_tfms]

    @torch.no_grad()
    def before_step(self, trainer: Trainer):
        xb, yb = to_device(trainer.batch, trainer.device)
        if trainer.training:
            xb = xb if self.x_tfm is None else self.x_tfm(xb)
            yb = yb if self.y_tfm is None else self.y_tfm(yb)
        else:
            xb = xb if self.x_tfms_valid is None else self.x_tfms_valid(xb)
            yb = yb if self.y_tfms_valid is None else self.y_tfms_valid(yb)
        for tfm in self.batch_tfms:
            xb, yb = tfm(xb, yb)
        trainer.batch = (xb, yb)
