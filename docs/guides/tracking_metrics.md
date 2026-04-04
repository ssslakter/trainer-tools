# Tracking Metrics

Metrics operate directly on the dictionary returned by your step functions (`trainer.state.output`). There are two ways to initialize metrics to access what they need.

## 1. Using dictionary keys (Default)
For the common case, simply provide the keys that your `train_step`/`eval_step` placed in the output dictionary:

```python
from trainer_tools.hooks import Accuracy, Loss

Accuracy(pred_key="logits", target_key="labels")
Loss(loss_key="loss") # Automatically logs the loss key
```

> **Note:** If metrics are requested but those keys are not mapped inside `trainer.state.output`, it will raise a `KeyError` giving clear indication it missed a key from the train/eval step output.

## 2. Using extractor lambdas (Advanced)
If your step output requires non-trivial processing (e.g., slicing, formatting, or extracting nested values) before comparing pred/labels, you can pass a custom function that takes the `trainer.state` object:

```python
from trainer_tools.hooks import Accuracy

def extract_preds(state):
    # Retrieve logits and apply argmax across embedding
    return state.output["logits"].argmax(dim=-1)

def extract_targets(state):
    return state.output["targets"]

Accuracy(
    name="my_accuracy",
    pred_fn=extract_preds,
    target_fn=extract_targets
)
```
