import time

from .base import Metric
from ...trainer import Trainer


class SamplesPerSecond(Metric):
    """Measures training throughput in samples per second.

    Uses wall-clock time between steps to compute throughput. The batch size
    is inferred from the current batch on first call, or can be provided
    explicitly via ``batch_size``.
    """

    def __init__(self, name="samples_per_second", freq=1, batch_size: int = None):
        super().__init__(name, freq, phase="after_step", use_prefix=False)
        self._batch_size = batch_size
        self._last_time: float = None

    def should_run(self, trainer):
        if not trainer.training:
            return False
        return super().should_run(trainer)

    def _infer_batch_size(self, batch):
        if isinstance(batch, (list, tuple)):
            return len(batch[0])
        if isinstance(batch, dict):
            return len(next(iter(batch.values())))
        return len(batch)

    def __call__(self, trainer: Trainer):
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return None

        elapsed = now - self._last_time
        self._last_time = now

        bs = self._batch_size or self._infer_batch_size(trainer.batch)
        return {self.name: bs / elapsed}
