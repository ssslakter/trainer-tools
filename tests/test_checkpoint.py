import torch, pytest
import torch.nn as nn
from trainer_tools.hooks.base import BaseHook
from trainer_tools.trainer import Trainer
from trainer_tools.hooks import CheckpointHook


def test_checkpoint_resume(simple_model, tuple_loaders, tmp_path):
    train_dl, valid_dl = tuple_loaders
    save_dir = tmp_path / "checkpoints"

    ckpt_hook_1 = CheckpointHook(save_dir=save_dir, save_every_steps=5, keep_last=5)

    class CancelHook(BaseHook):
        def __init__(self, step, epoch):
            self.step, self.epoch = step, epoch

        def after_step(self, trainer):
            if trainer.step >= self.step and trainer.epoch >= self.epoch:
                raise KeyboardInterrupt()

    trainer_1 = Trainer(
        model=simple_model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=torch.optim.Adam(simple_model.parameters()),
        loss_func=nn.CrossEntropyLoss(),
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
            assert trainer.epoch == 2
            assert trainer.step == 10
            w1 = simple_model.net.weight.detach()
            w2 = trainer.model.net.weight.detach()
            assert torch.equal(w1, w2)
            raise KeyboardInterrupt("Verification Complete")

    trainer_2 = Trainer(
        model=model_2,
        train_dl=train_dl,
        valid_dl=valid_dl,
        optim=torch.optim.Adam(model_2.parameters()),
        loss_func=nn.CrossEntropyLoss(),
        epochs=4,
        hooks=[ckpt_hook_2, VerificationHook()],
        device="cpu",
    )

    with pytest.raises(KeyboardInterrupt):
        trainer_2.fit()
