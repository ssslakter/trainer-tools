import logging
from .ema import EMAHook
from ..imports import *
from .base import BaseHook
from ..trainer import BaseTrainer
import os
from omegaconf import OmegaConf
from ..checkpoint import save_pretrained


log = logging.getLogger(__name__)


class CheckpointHook(BaseHook):
    """
    Saves model, optimizer, scheduler, scaler, and RNG states.
    Can resume training from a checkpoint.
    """

    ord = -50

    def __init__(
        self, save_dir: str, save_every_steps: int = 1000, keep_last: int = 3, resume_path: Optional[str] = None
    ):
        self.save_dir, self.every, self.keep_last = Path(save_dir), save_every_steps, keep_last
        self.resume_path = resume_path
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: list[Path] = []
        self.config_saved = False

    def _save(self, trainer: BaseTrainer, filename: str):
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

        # Save config
        if not self.config_saved and (config := getattr(trainer, "config", None)):
            config_path = self.save_dir / "config.yaml"
            with open(config_path, "w") as f:
                OmegaConf.save(config, f, resolve=True)
            log.info(f"Saved config: {config_path}")
            self.config_saved = True

        # Rotation logic
        if "interrupted" in filename:
            return

        self.saved_checkpoints.append(path)
        if len(self.saved_checkpoints) <= self.keep_last:
            return
        oldest = self.saved_checkpoints.pop(0)
        if oldest.exists():
            oldest.unlink()

    def before_fit(self, trainer: BaseTrainer):
        if not self.resume_path:
            return
        if Path(self.resume_path).exists():
            self.load_checkpoint(trainer, self.resume_path)
            log.info(f"Resumed training from checkpoint: {self.resume_path}, step {trainer.step}/{trainer.n_steps}")
        else:
            log.info(f"Resume path {self.resume_path} does not exist. Starting fresh training.")

    def after_step(self, trainer: BaseTrainer):
        if trainer.training and trainer.step > 0 and trainer.step % self.every == 0:
            self._save(trainer, f"checkpoint_step_{trainer.step}.pt")

    def after_cancel(self, trainer: BaseTrainer):
        self._save(trainer, "checkpoint_interrupted.pt")

    def after_fit(self, trainer: BaseTrainer):
        self._save(trainer, "model_final.pt")
        model_to_save = trainer.model
        if (ema_hook := trainer.get_hook(EMAHook, None)) and ema_hook.ema_model is not None:
            model_to_save = ema_hook.ema_model
            log.info("Using EMA model for pretrained export")

        save_pretrained(model_to_save, self.save_dir, config=getattr(trainer, "config", None))

    def load_checkpoint(self, trainer: BaseTrainer, path: str):
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
