import logging
from collections import defaultdict

from trainer_tools.utils import flatten_config
from ..imports import *
from .base import BaseHook

log = logging.getLogger(__name__)

try:
    import trackio as tio

    _HAS_TIO = True
except ImportError:
    tio = None
    _HAS_TIO = False

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False


class BaseMetricsHook(BaseHook):
    """An abstract hook to collect, aggregate, and plot training/validation metrics."""

    def __init__(self, verbose=True, tracker_type: str = None, config=None, **tracker_kwargs):
        self.verbose = verbose
        self.metrics = defaultdict(list)
        self.tracker = None
        self.use_tracker = False
        self.config = flatten_config(config) if config else None
        self.tracker_kwargs = tracker_kwargs
        
        if tracker_type == "trackio":
            if _HAS_TIO:
                self.tracker = tio
                self.use_tracker = True
            else:
                log.warning("'trackio' was requested but is not installed.")
        elif tracker_type == "wandb":
            if _HAS_WANDB:
                self.tracker = wandb
                self.use_tracker = True
            else:
                log.warning("'wandb' was requested but is not installed.")
        elif tracker_type is not None:
            raise ValueError(f"Unknown tracker_type: {tracker_type}. Choose from 'wandb', 'trackio', or None.")


    def _get_batch_metrics(self, trainer, prefix: str = Literal["train", "valid"]) -> dict:
        """Subclasses MUST implement this to return a dictionary of metrics for the current step."""
        raise NotImplementedError

    def before_fit(self, trainer):
        self.n_train, self.n_valid = len(trainer.train_dl), len(trainer.valid_dl)
        if self.use_tracker:
            self.tracker.init(config=self.config, **self.tracker_kwargs)

    def before_valid(self, trainer):
        self.val_batch_metrics = defaultdict(list)

    def after_loss(self, trainer):
        prefix = "train" if trainer.training else "valid"
        data = self._get_batch_metrics(trainer, prefix)

        metrics_dict = self.val_batch_metrics if not trainer.training else self.metrics
        for k, v in data.items():
            metrics_dict[k].append(v)
        if self.use_tracker and trainer.training:
            self.tracker.log(data, trainer.step)

    def after_epoch(self, trainer):
        for k, v in self.val_batch_metrics.items():
            self.metrics[k].append(np.mean(v))
        if self.use_tracker and (val_means := {k: self.metrics[k][-1] for k in self.val_batch_metrics}):
            self.tracker.log(val_means, trainer.step)
        train_loss = np.mean(self.metrics["train_loss"][-self.n_train :])
        val_loss = self.metrics["valid_loss"][-1]
        if self.verbose:
            log.info(f"Epoch {trainer.epoch+1}/{trainer.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def after_fit(self, trainer):
        if self.use_tracker:
            self.tracker.finish()

    def plot_loss(self, axes=None, from_step: int = 0):
        """Plots graphs for all available metrics (e.g., loss, kl, recon)."""
        train_keys = sorted([k for k in self.metrics if k.startswith("train_")])
        metrics_to_plot = [k.replace("train_", "") for k in train_keys]

        if not axes:
            _, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)), squeeze=False)
        axes = axes.flatten()

        for i, m in enumerate(metrics_to_plot):
            ax = axes[i]
            data = self.metrics[f"train_{m}"]
            ax.plot(list(range(from_step, len(data))), data[from_step:], label=f"Train {m.title()}")
            if f"valid_{m}" in self.metrics:
                epoch_from_step = from_step // self.n_train
                data = self.metrics[f"valid_{m}"]
                val_x = np.arange(1, len(data) + 1) * self.n_train - 1
                ax.plot(val_x[epoch_from_step:], data[epoch_from_step:], "o-", label=f"Valid {m.title()}")
            ax.set_title(m.replace("_", " ").title())
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Batch / Step")
        plt.tight_layout()
        return axes


class MetricsHook(BaseMetricsHook):
    def _get_batch_metrics(self, trainer, prefix):
        return {f"{prefix}_loss": trainer.loss}
