import pytest
import json
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TupleDataset(Dataset):
    """Generates simple (x, y) tuples."""

    def __init__(self, length=20):
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # input: size 10, target: class 0 or 1
        return torch.randn(10), torch.randint(0, 2, (1,)).item()


class DictDataset(Dataset):
    """Generates {'x': ..., 'labels': ...} dicts."""

    def __init__(self, length=20):
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"x": torch.randn(10), "labels": torch.tensor(torch.randint(0, 2, (1,)).item())}


class LinearDataset(Dataset):
    """
    Deterministic dataset for Linear Regression Metric Testing.
    Target y = 2 * x
    """

    def __init__(self):
        # simple inputs: 1.0, 2.0, 3.0, 4.0
        self.x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.y = self.x * 2.0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 2)

    def forward(self, x):
        return self.net(x)


class HFStyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 2)

    def forward(self, x, labels=None):
        logits = self.net(x)
        output = {"logits": logits}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            output["loss"] = loss_fct(logits, labels)
        return output


# --- Fixtures ---


@pytest.fixture
def simple_train_step():
    def train_step(batch, trainer):
        x, y = batch
        preds = trainer.model(x)
        loss_fn = nn.CrossEntropyLoss() if preds.shape[-1] > 1 else nn.MSELoss()

        # If the target is just [0.0] or something of shape (1, 1), MSELoss works best.
        # But here y might be target class index, so if preds shape is (..., 2) we do CE.
        if preds.shape[-1] == 2:
            loss = loss_fn(preds, y.squeeze() if y.ndim > 1 else y)
        else:
            loss = loss_fn(preds, y)
        return {"loss": loss, "preds": preds, "targets": y}

    return train_step


@pytest.fixture
def hf_train_step():
    def train_step(batch, trainer):
        # HFStyleModel expects kwargs
        outputs = trainer.model(**batch)
        if "labels" in batch:
            outputs["labels"] = batch["labels"]
        return outputs

    return train_step


@pytest.fixture
def tuple_loaders():
    ds = TupleDataset()
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


@pytest.fixture
def dict_loaders():
    ds = DictDataset()
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def hf_model():
    return HFStyleModel()
