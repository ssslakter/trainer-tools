from .utils import *
from tqdm.auto import tqdm


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: Callable, device=default_device, return_preds=False):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            xb, yb = to_device(batch, device)
            outputs = model(xb)
            preds.append(outputs.cpu())
            targets.append(yb.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)
    loss = criterion(preds, targets).item()
    if not return_preds:
        return loss
    return loss, {"preds": preds, "targets": targets}
