import torch
import logging
from pathlib import Path
from .base import BaseHook

log = logging.getLogger(__name__)

class MemoryProfilerHook(BaseHook):
    """
    PyTorch Memory Profiler Hook.
    
    Args:
        save_dir: Where to save the .pickle files.
        dump_every: Number of steps between snapshots.
        max_entries: Limit history size to avoid overhead (default 100,000).
        record_stacks: If 'all', captures stack traces for every allocation.
    """
    ord = 100 # Runs late to ensure all step processing is finished

    def __init__(
        self, 
        save_dir: str = "./memory_snapshots", 
        dump_every: int = 200, 
        max_entries: int = 100000,
        mode: str = "all"
    ):
        self.save_dir = Path(save_dir)
        self.dump_every = dump_every
        self.max_entries = max_entries
        self.mode = mode
        self._enabled = False

    def before_fit(self, trainer):
        if not torch.cuda.is_available():
            log.warning("MemoryProfilerHook: CUDA not available. Hook disabled.")
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.cuda.memory._record_memory_history(
                enabled=self.mode,
                max_entries=self.max_entries
            )
            self._enabled = True
            log.info(f"Memory history recording enabled (mode={self.mode}).")
        except Exception as e:
            log.error(f"Failed to start memory history recording: {e}")

    def after_step(self, trainer):
        if not self._enabled or not trainer.training:
            return

        if trainer.step > 0 and trainer.step % self.dump_every == 0:
            self._dump(f"step_{trainer.step}")

    def after_cancel(self, trainer):
        if self._enabled:
            self._dump("interrupted")
            self._disable()

    def after_fit(self, trainer):
        if self._enabled:
            self._dump("final")
            self._disable()

    def _dump(self, label: str):
        path = self.save_dir / f"snapshot_{label}.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(path))
            log.info(f"Memory snapshot saved: {path} -> Upload to https://pytorch.org/memory_viz")
        except Exception as e:
            log.error(f"Failed to dump memory snapshot: {e}")

    def _disable(self):
        torch.cuda.memory._record_memory_history(enabled=None)
        self._enabled = False
        log.info("Memory history recording disabled.")