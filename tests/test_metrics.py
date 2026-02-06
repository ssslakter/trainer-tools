import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import MetricsHook, Loss
from tests.conftest import LinearDataset


def test_mse_metric_adequacy():
    """
    Verifies that the loss metric is calculated mathematically correctly.

    Setup:
    - Data: x=[1,2,3,4], y=[2,4,6,8] (Target is 2x)
    - Model: Fixed to y = 3x (Bad model)

    Math:
    - Input 1 -> Pred 3 -> Target 2 -> MSE (3-2)^2 = 1
    - Input 2 -> Pred 6 -> Target 4 -> MSE (6-4)^2 = 4
    - Input 3 -> Pred 9 -> Target 6 -> MSE (9-6)^2 = 9
    - Input 4 -> Pred 12-> Target 8 -> MSE (12-8)^2 = 16

    Average MSE should be (1+4+9+16) / 4 = 30 / 4 = 7.5
    """

    ds = LinearDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=False)

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)

    metrics_hook = MetricsHook(metrics=[Loss()])

    trainer = Trainer(
        model=model,
        train_dl=dl,
        valid_dl=dl,
        optim=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_func=nn.MSELoss(),
        epochs=1,
        hooks=[metrics_hook],
        device="cpu",
    )

    trainer.fit()

    # Check Step-wise training loss
    # Batch 1 (x=1,2): Errors 1, 4 -> Mean 2.5
    # Batch 2 (x=3,4): Errors 9, 16 -> Mean 12.5
    train_losses = metrics_hook.history["train_loss"]
    assert len(train_losses) == 2
    assert abs(train_losses[0] - 2.5) < 1e-5
    assert abs(train_losses[1] - 12.5) < 1e-5

    # Check Epoch-wise validation loss
    # (2.5 + 12.5) / 2 = 7.5 OR (1+4+9+16)/4 = 7.5
    valid_losses = metrics_hook.history["valid_loss"]
    assert len(valid_losses) == 1
    assert abs(valid_losses[0] - 7.5) < 1e-5
