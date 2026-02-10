
import torch
import torch.nn as nn
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import GradientAccumulationHook
from torch.utils.data import DataLoader

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=False)
        self.fc.weight.data.fill_(1.0) # w = 1.0

    def forward(self, x):
        return self.fc(x)

def test_gradient_accumulation():
    torch.manual_seed(42)
    model = LinearModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    # x=1, y=0. batch_size=1
    x = torch.ones(1, 1)
    y = torch.zeros(1, 1)
    dataset = [(x, y) for _ in range(4)]
    dl = DataLoader(dataset, batch_size=1)

    accum_hook = GradientAccumulationHook(steps=2)

    trainer = Trainer(
        model=model,
        train_dl=dl,
        valid_dl=dl, 
        optim=opt,
        loss_func=nn.MSELoss(),
        epochs=1,
        hooks=[accum_hook],
        device="cpu"
    )

    # Theory:
    # MSE Loss L = (w*x - y)^2 = (w)^2 (since x=1, y=0)
    # dL/dw = 2w
    # Loss scaling: L' = L/2 => dL'/dw = w
    
    # Step 1: w=1.0. Grad = 1.0. Accumulate. w=1.0.
    # Step 2: w=1.0. Grad = 1.0 + 1.0 = 2.0. Update. w = 1.0 - 0.1 * 2.0 = 0.8. Zero grad.
    # Step 3: w=0.8. Grad = 0.8. Accumulate. w=0.8.
    # Step 4: w=0.8. Grad = 0.8 + 0.8 = 1.6. Update. w = 0.8 - 0.1 * 1.6 = 0.64. Zero grad.

    trainer.fit()

    final_w = model.fc.weight.item()
    print(f"Final weight: {final_w}")
    assert abs(final_w - 0.64) < 1e-5, f"Expected 0.64, got {final_w}"

if __name__ == "__main__":
    test_gradient_accumulation()
