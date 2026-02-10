from .base import BaseHook
from .metrics import *
from .pbar import ProgressBarHook
from .optimization import LRSchedulerHook, AMPHook, GradClipHook, GradientAccumulationHook
from .checkpoint import CheckpointHook
from .ema import EMAHook
from .transforms import BatchTransformHook
from ..utils import is_notebook

import matplotlib

if not is_notebook():
    matplotlib.use("Agg")