import logging
from torch.profiler import profile, ProfilerActivity, schedule
from .base import BaseHook
from ..imports import *

log = logging.getLogger(__name__)

class ProfilerHook(BaseHook):
    """
    A Hook to profile memory and performance using PyTorch Profiler.
    Optimized for finding RAM leaks.
    """
    ord = 100

    def __init__(
        self,
        skip_first: int = 5,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        profile_memory: bool = True,
        with_stack: bool = True,
        save_dir: str = "./profiler_logs",
        activities: Optional[List[ProfilerActivity]] = None
    ):
        self.save_dir = Path(save_dir)
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        
        self.activities = activities or [ProfilerActivity.CPU]
        if torch.cuda.is_available() and ProfilerActivity.CUDA not in self.activities:
            self.activities.append(ProfilerActivity.CUDA)

        self.schedule = schedule(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat
        )

    def _trace_handler(self, p):
        """This runs whenever a cycle of 'active' steps finishes."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        sort_by = "self_cpu_memory_usage" if self.profile_memory else "cpu_time_total"
        table = p.key_averages().table(sort_by=sort_by, row_limit=15)
        
        log.info(f"\n--- PROFILER REPORT (Step {p.step_num}) ---\n{table}")

        trace_path = self.save_dir / f"trace_step_{p.step_num}.json"
        p.export_chrome_trace(str(trace_path))
        log.info(f"Exported trace to {trace_path}")

        try:
            p.export_memory_timeline(str(self.save_dir / f"memory_step_{p.step_num}.html"))
        except Exception:
            pass

    def before_fit(self, trainer):
        self.prof = profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        )
        self.prof.__enter__()
        log.info("Profiler Hook started. Waiting for 'active' steps to report...")

    def after_step(self, trainer):
        if trainer.training:
            self.prof.step()

    def after_fit(self, trainer):
        self.prof.__exit__(None, None, None)
        log.info("Profiler Hook shut down.")