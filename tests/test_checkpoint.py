import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_tools.hooks.base import BaseHook
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import CheckpointHook


def test_periodic_checkpoint_matches_its_step(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    dl = DataLoader([(torch.ones(1), torch.zeros(1))] * 2, batch_size=1)

    def train_step(batch, trainer):
        x, y = batch
        return {"loss": nn.functional.mse_loss(trainer.model(x), y)}

    trainer = Trainer(
        model=model,
        train_dl=dl,
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        train_step=train_step,
        epochs=1,
        hooks=[CheckpointHook(save_dir=tmp_path, save_every_steps=1, keep_last=2)],
        device="cpu",
    )
    trainer.fit()

    first = torch.load(tmp_path / "checkpoint_step_1.pt", weights_only=False)
    assert first["optimizer_step"] == 1
    assert torch.allclose(first["model"]["weight"], torch.tensor([[0.8]]))


def test_checkpoint_replays_to_the_next_batch(tmp_path):
    def make_trainer(model, hooks):
        dl = DataLoader([(torch.ones(1), torch.zeros(1))] * 4, batch_size=1)

        def train_step(batch, trainer):
            x, y = batch
            return {"loss": nn.functional.mse_loss(trainer.model(x), y)}

        return Trainer(model, train_step, dl, optim=torch.optim.SGD(model.parameters(), lr=0.1), epochs=1, hooks=hooks, device="cpu")

    class StopAfterTwo(BaseHook):
        def after_step(self, trainer):
            if trainer.step_state.optimizer_step == 2:
                raise KeyboardInterrupt

    interrupted = nn.Linear(1, 1, bias=False)
    interrupted.weight.data.fill_(1.0)
    with pytest.raises(KeyboardInterrupt):
        make_trainer(interrupted, [CheckpointHook(tmp_path), StopAfterTwo()]).fit()

    resumed = nn.Linear(1, 1, bias=False)
    make_trainer(resumed, [CheckpointHook(tmp_path, resume_path=tmp_path / "checkpoint_interrupted.pt")]).fit()

    uninterrupted = nn.Linear(1, 1, bias=False)
    uninterrupted.weight.data.fill_(1.0)
    make_trainer(uninterrupted, []).fit()
    assert torch.equal(resumed.weight, uninterrupted.weight)


def test_checkpoint_restores_a_stateful_loader(tmp_path):
    class StatefulLoader:
        def __init__(self):
            self.pos = 0

        def __iter__(self):
            while self.pos < 4:
                self.pos += 1
                yield torch.ones(1, 1), torch.zeros(1, 1)

        def state_dict(self):
            return {"pos": self.pos}

        def load_state_dict(self, state):
            self.pos = state["pos"]

    def make_trainer(model, loader, hooks):
        def train_step(batch, trainer):
            x, y = batch
            return {"loss": nn.functional.mse_loss(trainer.model(x), y)}

        return Trainer(model, train_step, loader, optim=torch.optim.SGD(model.parameters(), lr=0.1), epochs=1, hooks=hooks)

    class StopAfterTwo(BaseHook):
        def after_step(self, trainer):
            if trainer.step_state.optimizer_step == 2:
                raise KeyboardInterrupt

    interrupted = nn.Linear(1, 1, bias=False)
    interrupted.weight.data.fill_(1.0)
    with pytest.raises(KeyboardInterrupt):
        make_trainer(interrupted, StatefulLoader(), [CheckpointHook(tmp_path), StopAfterTwo()]).fit()

    resumed = nn.Linear(1, 1, bias=False)
    make_trainer(resumed, StatefulLoader(), [CheckpointHook(tmp_path, resume_path=tmp_path / "checkpoint_interrupted.pt")]).fit()

    uninterrupted = nn.Linear(1, 1, bias=False)
    uninterrupted.weight.data.fill_(1.0)
    make_trainer(uninterrupted, StatefulLoader(), []).fit()
    assert torch.equal(resumed.weight, uninterrupted.weight)


def test_checkpoint_resume(simple_model, tuple_loaders, tmp_path, simple_train_step):
    train_dl, valid_dl = tuple_loaders
    save_dir = tmp_path / "checkpoints"

    ckpt_hook_1 = CheckpointHook(save_dir=save_dir, save_every_steps=5, keep_last=5)

    class CancelHook(BaseHook):
        def __init__(self, step, epoch):
            self.step, self.epoch = step, epoch

        def after_step(self, trainer):
            if trainer.step_state.optimizer_step >= self.step and trainer.step_state.epoch >= self.epoch:
                raise KeyboardInterrupt()

    trainer_1 = Trainer(
        model=simple_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=torch.optim.Adam(simple_model.parameters()),
        train_step=simple_train_step,
        epochs=4,
        hooks=[ckpt_hook_1, CancelHook(step=10, epoch=2)],
        device="cpu",
    )
    with pytest.raises(KeyboardInterrupt):
        trainer_1.fit()

    expected_ckpt = save_dir / "checkpoint_interrupted.pt"
    assert expected_ckpt.exists()

    model_2 = type(simple_model)()
    ckpt_hook_2 = CheckpointHook(save_dir=save_dir, resume_path=str(expected_ckpt), save_every_steps=5)

    class VerificationHook(BaseHook):
        ord = 200
        """Checks state immediately after loading, before training moves on."""

        def before_fit(self, trainer):
            assert trainer.step_state.epoch == 2
            assert trainer.step_state.optimizer_step == 11
            w1 = simple_model.net.weight.detach()
            w2 = trainer.model.net.weight.detach()
            assert torch.equal(w1, w2)
            raise KeyboardInterrupt("Verification Complete")

    trainer_2 = Trainer(
        model=model_2,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=torch.optim.Adam(model_2.parameters()),
        train_step=simple_train_step,
        epochs=4,
        hooks=[ckpt_hook_2, VerificationHook()],
        device="cpu",
    )

    with pytest.raises(KeyboardInterrupt):
        trainer_2.fit()
