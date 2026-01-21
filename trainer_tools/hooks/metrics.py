import logging
from collections import defaultdict
from trainer_tools.utils import flatten_config
from ..imports import *
from .base import BaseHook
from ..metrics import Metric, FunctionalMetric

log = logging.getLogger(__name__)

try:
    import trackio as tio

    _HAS_TIO = True
except ImportError:
    tio, _HAS_TIO = None, False
try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    wandb, _HAS_WANDB = None, False


class LoggerHook(BaseHook):
    """
    Aggregates data from multiple Metrics and logs to console/tracker.
    Only ONE instance of this hook is needed per Trainer.
    """

    def __init__(
        self,
        metrics: List[Union[Metric, Callable]],
        verbose=True,
        tracker_type: str = None,
        config=None,
        **tracker_kwargs,
    ):
        self.verbose, self.config, self.tracker_kwargs = verbose, config, tracker_kwargs

        self.metrics = [m if isinstance(m, Metric) else FunctionalMetric(m) for m in metrics]
        self._phases: dict[str, list[Metric]] = defaultdict(list)
        for m in self.metrics:
            self._phases[m.phase].append(m)

        # Buffers
        self.step_data = {}
        self.epoch_data = defaultdict(list)

        # History now tracks values AND steps separately to handle freq > 1
        self.history = defaultdict(list)  # {'train_loss': [0.5, 0.4...]}
        self.history_steps = defaultdict(list)  # {'train_loss': [10, 20...]}

        self.steps_per_epoch = 0
        self._init_tracker(tracker_type)

    def _init_tracker(self, t_type):
        self.tracker, self.use_tracker = None, False
        if t_type == "trackio" and _HAS_TIO:
            self.tracker, self.use_tracker = tio, True
        elif t_type == "wandb" and _HAS_WANDB:
            self.tracker, self.use_tracker = wandb, True
        elif t_type:
            log.warning(f"Tracker '{t_type}' not found.")

    def _run_metrics(self, trainer, phase):
        prefix = "train" if trainer.training else "valid"

        for m in self._phases[phase]:
            if not m.should_run(trainer):
                continue

            data = m(trainer)
            if not data:
                continue
            p_data = {f"{prefix}_{k}": v for k, v in data.items()}
            self.step_data.update(p_data)

            for k, v in p_data.items():
                self.epoch_data[k].append(v)
                if trainer.training:
                    self.history[k].append(v)
                    self.history_steps[k].append(trainer.step)

    def before_fit(self, trainer):
        self.steps_per_epoch = len(trainer.train_dl)
        if self.use_tracker:
            self.tracker.init(config=self.config, **self.tracker_kwargs)

    def before_epoch(self, trainer):
        self.epoch_data.clear()

    def after_pred(self, trainer):
        self._run_metrics(trainer, "after_pred")

    def after_loss(self, trainer):
        self._run_metrics(trainer, "after_loss")

    def after_backward(self, trainer):
        self._run_metrics(trainer, "after_backward")

    def after_step(self, trainer):
        self._run_metrics(trainer, "after_step")
        if trainer.training and self.use_tracker and self.step_data:
            self.tracker.log(self.step_data, trainer.step)
        self.step_data.clear()

    def after_epoch(self, trainer):
        val_stats = {k: np.mean(v) for k, v in self.epoch_data.items() if k.startswith("valid_")}

        for k, v in val_stats.items():
            self.history[k].append(v)
            self.history_steps[k].append((trainer.epoch + 1) * self.steps_per_epoch)

        if self.use_tracker and val_stats:
            self.tracker.log(val_stats, trainer.step)

        if self.verbose:
            logs = [f"Epoch {trainer.epoch+1}/{trainer.epochs}"]
            # Find latest train stats (handling different frequencies)
            unique_metrics = set(k.split("_", 1)[1] for k in self.history if k.startswith("train_"))
            for base_k in sorted(unique_metrics):
                k = f"train_{base_k}"
                if self.history[k]:
                    # Simple average of last few collected points
                    logs.append(f"T_{base_k}: {np.mean(self.history[k][-10:]):.3f}")

            for k, v in val_stats.items():
                logs.append(f"V_{k[6:]}: {v:.3f}")
            log.info(" | ".join(logs))

    def after_fit(self, trainer):
        if self.use_tracker:
            self.tracker.finish()

    def plot(self, axes=None, metrics: List[str] = ["loss"]):
        """
        Plots collected metrics.

        Args:
            axes: Optional Matplotlib axes.
            metrics: List of base metric names to plot (e.g. ["loss", "acc"]).
                     Defaults to ["loss"]. If set to None, plots ALL collected metrics.
        """

        available = sorted(list(set(k.split("_", 1)[1] for k in self.history if k.startswith("train_"))))
        keys = available if metrics is None else [k for k in metrics if k in available]

        if not keys:
            log.warning("No matching metrics found to plot.")
            return

        if axes is None:
            fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), sharex=True)
            if len(keys) == 1:
                axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, key in zip(axes, keys):
            # Train: Use recorded steps for X-axis
            if t_y := self.history.get(f"train_{key}"):
                t_x = self.history_steps.get(f"train_{key}")
                ax.plot(t_x, t_y, label=f"Train {key}")

            # Valid: Use recorded steps (epoch boundaries)
            if v_y := self.history.get(f"valid_{key}"):
                v_x = self.history_steps.get(f"valid_{key}")
                ax.plot(v_x, v_y, "o-", label=f"Valid {key}")

            ax.set_ylabel(key.title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        if axes:
            axes[-1].set_xlabel("Steps")
        plt.tight_layout()
