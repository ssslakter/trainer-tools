import logging, json
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


class MetricsHook(BaseHook):
    """
    Aggregates data from multiple Metrics and logs to console/tracker.
    Only ONE instance of this hook is needed per Trainer.
    """
    ord = -10

    def __init__(
        self,
        metrics: List[Metric],
        verbose=True,
        tracker_type: str = None,
        config: Union[dict, str] = None,
        **tracker_kwargs,
    ):
        self.verbose, self.tracker_kwargs = verbose, tracker_kwargs
        self.config = flatten_config(json.loads(config) if isinstance(config, str) else config)

        self.metric_types = metrics
        self._phases: dict[str, list[Metric]] = defaultdict(list)
        for m in self.metric_types:
            self._phases[m.phase].append(m)

        # Buffers & History
        self.step_data = {}  # Current step buffer
        self.epoch_data = defaultdict(list)  # Accumulator for epoch averages
        self.history = defaultdict(list)
        self.history_steps = defaultdict(list)

        self.steps_per_epoch = 0
        self._init_tracker(tracker_type)

    @property
    def metrics(self):
        return self.history

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
            p_data = data if not m.use_prefix else {f"{prefix}_{k}": v for k, v in data.items()}
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

        logs = [f"Epoch {trainer.epoch+1}/{trainer.epochs}"]

        for k in sorted(self.history.keys()):
            if "valid_" not in k and self.history[k] and (self.verbose or "loss" in k.lower()):
                val = np.mean(self.history[k][-10:])  # smooth last 10 steps
                logs.append(f"{k}: {val:.3f}")

        for k, v in val_stats.items():
            if self.verbose or "loss" in k.lower():
                logs.append(f"{k}: {v:.3f}")

        log.info(" | ".join(logs))

    def after_fit(self, trainer):
        if self.use_tracker:
            self.tracker.finish()

    def plot(self, axes=None, metrics: Optional[List[str]] = ["loss"]):
        """
        Plots metrics. Handles both prefixed (train_loss/valid_loss) and raw (grad_norm) keys.
        If metrics is None, plots ALL available keys.
        """
        all_keys = set()
        for k in self.history.keys():
            pre = "train_"
            if k.startswith(pre):
                all_keys.add(k[len(pre) :])
            elif not k.startswith("valid_"):
                all_keys.add(k)

        keys = sorted(list(all_keys)) if metrics is None else [k for k in metrics if k in all_keys]
        if not keys:
            return

        if axes is None:
            fig, axes = plt.subplots(len(keys), 1, figsize=(7, 3 * len(keys)), sharex=True)
            if len(keys) == 1:
                axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, key in zip(axes, keys):
            t_key = f"train_{key}" if f"train_{key}" in self.history else key

            if t_key in self.history:
                ax.plot(self.history_steps[t_key], self.history[t_key], label=f"Train {key}")

            v_key = f"valid_{key}"
            if v_key in self.history:
                ax.plot(self.history_steps[v_key], self.history[v_key], "o-", label=f"Valid {key}")

            ax.set_ylabel(key.title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        if axes:
            axes[-1].set_xlabel("Steps")
        plt.tight_layout()
