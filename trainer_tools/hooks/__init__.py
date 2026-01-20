from .base import BaseHook
from .metrics import MetricsHook
from .metrics_diffusion import DiffusionMetricsHook
from .pbar import ProgressBarHook
from .optimization import LRSchedulerHook, AMPHook, GradClipHook
from .checkpoint import CheckpointHook
from .sampling import *
from .text import *
from .ema import EMAHook
from .toy import ToyMetricsHook, SequenceSamplingHook
from ...utils import is_notebook

import matplotlib

if not is_notebook():
    matplotlib.use("Agg")