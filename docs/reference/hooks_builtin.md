# Built-in Hooks

## Base Hooks

::: trainer_tools.hooks.BaseHook
    options:
      members:
        - before_fit
        - before_epoch
        - before_step
        - after_pred
        - after_loss
        - after_backward
        - after_step
        - before_valid
        - after_epoch
        - after_fit
        - after_cancel

::: trainer_tools.hooks.MainProcessHook

::: trainer_tools.hooks.LambdaHook

## Optimization Hooks

::: trainer_tools.hooks.AMPHook
    options:
      members:
        - __init__

::: trainer_tools.hooks.GradientAccumulationHook
    options:
      members:
        - __init__

::: trainer_tools.hooks.GradClipHook
    options:
      members:
        - __init__

::: trainer_tools.hooks.LRSchedulerHook
    options:
      members:
        - __init__

## Checkpointing and Utilities

::: trainer_tools.hooks.CheckpointHook
    options:
      members:
        - __init__

::: trainer_tools.hooks.ProgressBarHook
    options:
      members:
        - __init__
