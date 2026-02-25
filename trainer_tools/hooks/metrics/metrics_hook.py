import logging, json
from collections import defaultdict
from trainer_tools.utils import flatten_config
from ...imports import *
from ...hooks.base import BaseHook
from .base import Metric

log = logging.getLogger(__name__)

_TRACKERS = {}
try:
    import trackio as tio
    import os

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    _TRACKERS["trackio"] = tio
except ImportError:
    pass
try:
    import wandb

    _TRACKERS["wandb"] = wandb
except ImportError:
    pass
try:
    import mlop

    _TRACKERS["mlop"] = mlop
except ImportError:
    pass


class MetricsHook(BaseHook):
    """
    Aggregates data from multiple Metrics and logs to console/tracker.
    Only ONE instance of this hook is needed per Trainer.
    """

    ord = -10

    def __init__(
        self,
        metrics: List[Metric],
        verbose: bool = True,
        tracker_type: str = None,
        config: Union[dict, str] = None,
        freq: int = 1,
        log_file: Optional[str] = "metrics.jsonl",
        name: str = None,
        project: str = None,
        **tracker_kwargs,
    ):
        """Aggregates Metric outputs and logs them to the console and/or a tracker.

        Args:
            metrics: List of :class:`Metric` instances to evaluate and aggregate.
            verbose: If ``True``, all metric keys are printed after each epoch.
                If ``False``, only keys containing ``"loss"`` are printed.
            tracker_type: Backend to use for logging. Supported values are
                ``"wandb"``, ``"trackio"``, ``"mlop"``, and ``"file"``.
                ``"file"`` writes metrics to *log_file* as JSONL.
                ``None`` logs to stdout only (via the standard logger).
            config: Hyperparameter config forwarded to the tracker on init.
                Accepts a ``dict`` or a JSON-encoded string.
            freq: Step interval at which step-level metrics are logged.
            log_file: Path to the JSONL file used when ``tracker_type`` is
                ``"file"`` (or ``None``). Defaults to ``"metrics.jsonl"``.
            name: Run name forwarded to the tracker. Convenience alias for
                passing ``name`` via *tracker_kwargs*.
            project: Project name forwarded to the tracker. Convenience alias
                for passing ``project`` via *tracker_kwargs*.
            **tracker_kwargs: Additional keyword arguments passed to the
                tracker's ``init`` call.
        """
        if name is not None:
            tracker_kwargs.setdefault("name", name)
        if project is not None:
            tracker_kwargs.setdefault("project", project)
        self.verbose, self.tracker_kwargs = verbose, tracker_kwargs
        self.config = flatten_config(json.loads(config) if isinstance(config, str) else config or {})
        self.freq, self.log_file = freq, Path(log_file) if log_file else None

        self.metric_types = metrics
        self._phases: dict[str, list[Metric]] = defaultdict(list)
        for m in self.metric_types:
            self._phases[m.phase].append(m)

        # Buffers & History
        self.step_data = {}
        self.epoch_data = {}
        self.aggregators = defaultdict(float)
        self.counts = defaultdict(int)
        self._init_tracker(tracker_type)

    def _init_tracker(self, t_type):
        self.tracker, self.use_tracker, self.use_file = None, False, False
        if t_type is None:
            return
        if t_type == "file":
            self.use_file = True
            if self.log_file and self.log_file.is_file():
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                self.log_file.unlink(missing_ok=True)
            return
        if t_type not in _TRACKERS:
            log.warning(f"Tracker '{t_type}' not found or not installed.")
            return
        self.tracker, self.use_tracker = _TRACKERS[t_type], True
        if t_type == "trackio":
            self.tracker_kwargs.setdefault("embed", False)  # improves performance

    def _run_metrics(self, trainer, phase):
        prefix = "train" if trainer.training else "valid"
        for m in self._phases[phase]:
            if not m.should_run(trainer):
                continue

            data = m(trainer)
            if not data:
                continue
            p_data = data if not m.use_prefix else {f"{prefix}_{k}": v for k, v in data.items()}

            for k, v in p_data.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy().tolist()

                self.step_data[k] = v
                if isinstance(v, (int, float)):
                    self.aggregators[k] += v
                    self.counts[k] += 1

    def before_fit(self, trainer):
        dl = getattr(trainer, "dl", getattr(trainer, "train_dl"))
        self.steps_per_epoch = len(dl)
        if self.use_tracker and trainer.is_main:
            self.tracker.init(config=self.config, **self.tracker_kwargs)

    def before_epoch(self, trainer):
        self.aggregators.clear()
        self.counts.clear()

    def after_pred(self, trainer):
        self._run_metrics(trainer, "after_pred")

    def after_loss(self, trainer):
        self._run_metrics(trainer, "after_loss")

    def after_backward(self, trainer):
        self._run_metrics(trainer, "after_backward")

    def after_step(self, trainer):
        self._run_metrics(trainer, "after_step")
        self.step_data["step"] = trainer.step
        if trainer.training and trainer.step % self.freq == 0 and trainer.is_main:
            if self.use_tracker:
                current_step = self.step_data.pop("step", trainer.step)
                self.tracker.log(self.step_data, current_step)
            elif self.use_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(self.step_data) + "\n")
        self.step_data.clear()

    def after_epoch(self, trainer):
        self.epoch_data = epoch_means = {k: self.aggregators[k] / self.counts[k] for k in self.aggregators}
        val_stats = {k: v for k, v in epoch_means.items() if k.startswith("valid_")}

        if trainer.is_main:
            if self.use_tracker and val_stats:
                self.tracker.log(val_stats, trainer.step)
            elif self.use_file and val_stats:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps({"epoch": trainer.epoch, **epoch_means}) + "\n")

        logs = [f"Epoch {trainer.epoch+1}/{trainer.epochs}"]

        for k in sorted(epoch_means.keys()):
            if self.verbose or "loss" in k.lower():
                logs.append(f"{k}: {epoch_means[k]:.4f}")

        if trainer.is_main:
            log.info(" | ".join(logs))

    def after_fit(self, trainer):
        if self.use_tracker and trainer.is_main:
            self.tracker.finish()

    def plot(self, axes=None, metrics=["loss"], show_epochs=False):
        self.history = load_metrics(self.log_file)
        all_roots = {k.replace("train_", "").replace("valid_", "") for cat in self.history for k in self.history[cat]}
        keys = sorted(all_roots) if metrics is None else [m for m in metrics if m in all_roots]

        if not keys:
            return

        if axes is None:
            fig, axes = plt.subplots(len(keys), 1, figsize=(8, 4 * len(keys)))
        axes = np.atleast_1d(axes)

        steps_per_epoch = getattr(self, "steps_per_epoch", None)

        for ax, root in zip(axes, keys):
            for cat in ["step", "epoch"]:
                if cat == "epoch" and not show_epochs:
                    continue
                for pre in ["train_", "valid_", ""]:
                    full_key = f"{pre}{root}"
                    if full_key in self.history[cat]:
                        data = self.history[cat][full_key]
                        if cat == "epoch" and steps_per_epoch is not None:
                            data = data.copy()
                            data[0] = (data[0] + 1) * steps_per_epoch
                        fmt = "o-" if "valid" in pre or cat == "epoch" else "-"
                        ax.plot(*data, fmt, label=f"{cat} {pre}{root}".strip())

            ax.set_ylabel(root.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()


def load_metrics(path):
    raw, res = defaultdict(list), {"step": {}, "epoch": {}}
    with open(path) as f:
        for d in (json.loads(l) for l in f if l.strip()):
            k = "step" if "step" in d else "epoch"
            idx = d.pop(k)
            for metric, val in d.items():
                raw[k, metric].append([idx, val])

    for (k, metric), v in raw.items():
        res[k][metric] = np.array(v, dtype=np.float32).T
    return res
