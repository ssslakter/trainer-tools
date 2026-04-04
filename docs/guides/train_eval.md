# The `train_step` and `eval_step` Callbacks

The `Trainer` requires a `train_step` callback to define how a single batch is processed. The function receives the current `batch` and the `trainer` instance, and **must return a dictionary** containing at least a `"loss"` key (a scalar or a 0d tensor) for the backward pass. The same dictionary is automatically exposed to metrics and other hooks via `trainer.state.output`.

## Example: A Basic Training Step

```python
def train_step(batch, trainer):
    inputs, labels = batch
    logits = trainer.model(inputs)
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Must return a dictionary containing at least the "loss" key!
    return {
        "loss": loss,
        "logits": logits,
        "labels": labels
    }
```

## Custom Evaluation Logic

By default, the `Trainer` uses the same `train_step` for validation. However, if your evaluation logic differs (e.g., using beam search for text generation, caching hidden states safely, or returning different dictionary keys format), you can provide an optional `eval_step` parameter:

```python
def eval_step(batch, trainer):
    inputs, labels = batch
    # Custom evaluation logic, e.g., generation
    generated_tokens = trainer.model.generate(inputs)
    
    return {
        "loss": 0.0, # Optional if running only inference, but good to include as dummy if needed by others
        "preds": generated_tokens,
        "targets": labels
    }

trainer = Trainer(
    model=model,
    train_step=train_step,
    eval_step=eval_step,
    # ...
)
```
