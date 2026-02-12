from .utils import *
from tqdm.auto import tqdm
from .trainer import Trainer


def evaluate(trainer: Trainer, dataloader: DataLoader, return_preds=False):
    trainer.model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = to_device(batch, trainer.device)
            yb = trainer.get_target(batch)
            outputs = trainer.predict(batch)
            preds.append(to_device(outputs, "cpu"))
            targets.append(to_device(yb, "cpu"))

    def _concat(samples):
        if not samples:
            return None
        if isinstance(samples[0], torch.Tensor):
            return torch.cat(samples)
        if isinstance(samples[0], dict):
            return {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        return samples

    preds, targets = _concat(preds), _concat(targets)
    loss = trainer.get_loss(preds, targets).item()
    if not return_preds:
        return loss
    return loss, {"preds": preds, "targets": targets}
