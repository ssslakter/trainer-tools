import logging
from .ema import EMAHook
from ..imports import *
from .base import BaseHook
from ..trainer import Trainer
import os
from omegaconf import OmegaConf
from ..checkpoint import save_pretrained
from .metrics import MetricsHook

log = logging.getLogger(__name__)


class CheckpointHook(BaseHook):
    """
    Saves model, optimizer, scheduler, scaler, and RNG states.
    Can resume training from a checkpoint.
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 1000,
        keep_last: int = 3,
        resume_path: Optional[str] = None,
        save_strategy: Literal["best", "latest"] = "best",
        metric_name: str = "valid_loss",
    ):
        self.save_dir, self.every, self.keep_last = Path(save_dir), save_every_steps, keep_last
        self.resume_path, self.save_strategy, self.metric = resume_path, save_strategy, metric_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: list[Path] = []
        self.config_saved = False

    def _save_config(self, trainer: Trainer):
        if self.config_saved:
            return
        if not (config := getattr(trainer, "config", None)):
            return
        config_path = self.save_dir / "config.yaml"
        OmegaConf.save(config, config_path, resolve=True)
        log.info(f"Saved config: {config_path}")
        self.config_saved = True

    def _save(self, trainer: Trainer, filename: str):
        path = self.save_dir / filename
        state = {
            "model": trainer.model.state_dict(),
            "opt": trainer.opt.state_dict(),
            "epoch": trainer.epoch,
            "step": trainer.step,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        # Save Scaler state if AMP is used
        if hasattr(trainer, "scaler"):
            state["scaler"] = trainer.scaler.state_dict()

        # Save Scheduler state if LRSchedulerHook exists
        from .optimization import LRSchedulerHook

        if sched_hook := trainer.get_hook(LRSchedulerHook, None):
            state["scheduler"] = sched_hook.sched.state_dict()

        if (ema_hook := trainer.get_hook(EMAHook, None)) and ema_hook.ema_model is not None:
            state["ema"] = ema_hook.ema_model.state_dict()

        torch.save(state, path)
        log.info(f"Saved checkpoint: {path}")

        # Rotation logic
        if "interrupted" in filename:
            return

        self.saved_checkpoints.append(path),
        if len(self.saved_checkpoints) <= self.keep_last:
            return
        oldest = self.saved_checkpoints.pop(0)
        if oldest.exists():
            oldest.unlink()

    def before_fit(self, trainer: Trainer):
        self._save_config(trainer)

        if not self.resume_path:
            return
        if Path(self.resume_path).exists():
            self.load_checkpoint(trainer, self.resume_path)
            log.info(f"Resumed training from checkpoint: {self.resume_path}, step {trainer.step}/{trainer.n_steps}")
        else:
            log.info(f"Resume path {self.resume_path} does not exist. Starting fresh training.")

    def after_step(self, trainer: Trainer):
        if not (trainer.training and trainer.step > 0 and trainer.step % self.every == 0):
            return
        if self.save_strategy == "latest":
            self._save(trainer, f"checkpoint_step_{trainer.step}.pt")
        elif self.save_strategy == "best":
            metrics_hook = trainer.get_hook(MetricsHook, None)
            metrics = metrics_hook.metrics.get(self.metric, []) if metrics_hook else []
            if not metrics:
                log.warning(f"No MetricsHook or metric '{self.metric}' found, switching save_strategy to 'latest'.")
                self.save_strategy = "latest"
                self._save(trainer, f"checkpoint_step_{trainer.step}.pt")
                return
            m = metrics[-1]
            if not hasattr(self, '_best_metric') or m < self._best_metric:
                self._best_metric = m
                self._save(trainer, f"checkpoint_best_{trainer.step}.pt")

    def after_cancel(self, trainer: Trainer):
        self._save(trainer, "checkpoint_interrupted.pt")

    def after_fit(self, trainer: Trainer):
        self._save(trainer, "model_final.pt")
        model_to_save = trainer.model
        if (ema_hook := trainer.get_hook(EMAHook, None)) and ema_hook.ema_model is not None:
            model_to_save = ema_hook.ema_model
            log.info("Using EMA model for pretrained export")

        save_pretrained(model_to_save, self.save_dir, config=getattr(trainer, "config", None))

    def load_checkpoint(self, trainer: Trainer, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        log.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)

        trainer.model.load_state_dict(checkpoint["model"])
        trainer.opt.load_state_dict(checkpoint["opt"])
        trainer.epoch = checkpoint["epoch"]
        trainer.step = checkpoint["step"]

        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["rng_cuda"].cpu())
        np.random.set_state(checkpoint["rng_numpy"])

        if "scaler" in checkpoint and hasattr(trainer, "scaler"):
            trainer.scaler.load_state_dict(checkpoint["scaler"])

        if "scheduler" in checkpoint:
            from .optimization import LRSchedulerHook

            try:
                trainer.get_hook(LRSchedulerHook).sched.load_state_dict(checkpoint["scheduler"])
            except KeyError:
                log.warning("Checkpoint has scheduler state but no LRSchedulerHook found.")
        if "ema" in checkpoint:
            trainer._ema_state_buffer = checkpoint["ema"]
        log.info(f"Resumed at Epoch {trainer.epoch}, Step {trainer.step}")
