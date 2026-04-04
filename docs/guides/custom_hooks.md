# Writing Custom Hooks

`trainer-tools` relies on `BaseHook` running at specific lifecycle points inside the `Trainer` class. You can create custom behavior by subclassing it and defining methods that run at those specific lifecycle points (`before_step`, `after_backward`, `after_epoch`, etc.).

## Example: Capturing Backward Gradients

Here is an example of creating a custom hook that collects the mean gradient magnitude of a specific model layer from the model, and exposes it temporarily to the trainer state's output so it can log automatically, while accumulating history manually!

```py
from trainer_tools.hooks import BaseHook

class GradientMonitorHook(BaseHook):
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.grad_history = []

    def after_backward(self, trainer):
        # We only collect gradients during training runs (where backprop happens),
        # validation steps do not do a backward pass!
        if not trainer.model.training:
            return
            
        # Inspect model and get the gradient
        for name, param in trainer.model.named_parameters():
            if name == self.layer_name and param.grad is not None:
                # Capture mean magnitude
                mean_grad = param.grad.abs().mean().item()
                self.grad_history.append(mean_grad)
                
                # We can also manipulate trainer state!
                # Let's log it into the output dict so other hooks (like metrics!) can see it!
                trainer.state.output[f"grad_{name}"] = mean_grad
```

## Understanding the Lifecycle

`Trainer.fit()` runs a standard PyTorch training loop but exposes **hook points** at every meaningful stage so you can inject logic without touching the core loop.

The execution order is roughly:

```text
fit()
├── before_fit
└── for each epoch:
    ├── before_epoch
    ├── for each training batch:
    │   ├── before_step
    │   ├── train_step()      → populates trainer.state.output
    │   ├── do_backward()     → loss_t.backward()
    │   ├── after_backward
    │   ├── do_opt_step()     → opt.step()
    │   ├── do_zero_grad()    → opt.zero_grad()
    │   └── after_step
    ├── before_valid
    ├── for each validation batch:
    │   └── (same as above, invoking eval_step() without backward/opt steps)
    └── after_epoch
└── after_fit           (or after_cancel on KeyboardInterrupt)
```

## Exploring Hook Order

A quick way to understand exactly when and what lifecycle hooks trigger depending where you place them is to utilize the `trainer.describe_hooks()` command on an instantiated trainer.

```python
trainer.describe_hooks()
```

It prints exactly what hooks have connected across the lifecycle endpoints!
