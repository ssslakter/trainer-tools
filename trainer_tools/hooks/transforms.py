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
        trainer.batch = to_device(trainer.batch, trainer.device)
        xb = trainer.get_input(trainer.batch)
        yb = trainer.get_target(trainer.batch)

        if trainer.training:
            if self.x_tfm is not None:
                xb = self.x_tfm(xb)
            if self.y_tfm is not None:
                yb = self.y_tfm(yb)
        else:
            if self.x_tfms_valid is not None:
                xb = self.x_tfms_valid(xb)
            if self.y_tfms_valid is not None:
                yb = self.y_tfms_valid(yb)

        if isinstance(trainer.batch, (list, tuple)):
            trainer.batch = (xb, yb)
        elif isinstance(trainer.batch, dict):
            trainer.batch.update(yb)
            # update xb after yb in case they return the same dict
            trainer.batch.update(xb)
        else:
            trainer.batch = xb

        for tfm in self.batch_tfms:
            trainer.batch = tfm(trainer.batch)
