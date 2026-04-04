# Trainer API

::: trainer_tools.trainer.Trainer
    handler: python
    options:
      members:
        - fit
        - do_backward
        - do_opt_step
        - do_zero_grad
        - describe_hooks
        - get_hook

::: trainer_tools.trainer.StepState
    handler: python
    options:
      members:
        - should_step_optimizer
        - increment_batch
        - reset_epoch
