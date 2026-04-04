import torch
import torch.nn as nn
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import MetricsHook, Accuracy, Loss, load_metrics


def test_trainer_tuple_dataloader(simple_model, tuple_loaders, simple_train_step):
    train_dl, valid_dl = tuple_loaders
    opt = torch.optim.Adam(simple_model.parameters(), lr=0.01)

    trainer = Trainer(
        model=simple_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt,
        train_step=simple_train_step,
        epochs=1,
        device="cpu",
    )

    trainer.fit()

    assert trainer.state.optimizer_step > 0


def test_trainer_dict_dataloader(hf_model, dict_loaders, tmp_path, hf_train_step):
    train_dl, valid_dl = dict_loaders
    opt = torch.optim.Adam(hf_model.parameters(), lr=0.01)
    hist_file = tmp_path / "metrics.jsonl"
    metrics = MetricsHook(
        metrics=[Accuracy(pred_key="logits", target_key="labels"), Loss()], log_file=hist_file, tracker_type="file"
    )

    trainer = Trainer(
        model=hf_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=opt,
        train_step=hf_train_step,
        epochs=1,
        hooks=[metrics],
        device="cpu",
    )
    trainer.fit()
    history = load_metrics(hist_file)
    assert "train_loss" in history["step"]
