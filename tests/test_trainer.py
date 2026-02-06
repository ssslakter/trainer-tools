import torch
import torch.nn as nn
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import MetricsHook, Accuracy, Loss


def test_trainer_tuple_dataloader(simple_model, tuple_loaders):
    train_dl, valid_dl = tuple_loaders
    opt = torch.optim.Adam(simple_model.parameters(), lr=0.01)

    trainer = Trainer(
        model=simple_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt,
        loss_func=nn.CrossEntropyLoss(),
        epochs=1,
        device="cpu",
    )

    trainer.fit()

    assert trainer.step > 0
    # # Memory leak check: attributes should be cleared
    # assert trainer.preds is None
    # assert trainer.loss_t is None


def test_trainer_dict_dataloader(hf_model, dict_loaders):
    train_dl, valid_dl = dict_loaders
    opt = torch.optim.Adam(hf_model.parameters(), lr=0.01)

    metrics = MetricsHook(metrics=[Accuracy(), Loss()])

    def loss_fn(preds, target):
        return torch.nn.functional.cross_entropy(preds["logits"], target)

    trainer = Trainer(
        model=hf_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt,
        epochs=1,
        loss_func=loss_fn,
        hooks=[metrics],
        device="cpu",
    )

    trainer.fit()
    assert "train_loss" in metrics.history
