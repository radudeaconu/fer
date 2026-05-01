"""Smoke tests for src.train and src.eval against a tiny dummy model on synthetic data.

We don't import DAN/POSTER++ here (those need third_party/ cloned). Instead we
test the model-agnostic plumbing: dataloaders, train_one_epoch, evaluate,
confusion_matrix, per_class_metrics. If these pass on a Linear classifier,
they'll pass on the real models — only the model factory differs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.data import (
    CLASSES,
    NUM_CLASSES,
    FER2013Dataset,
    build_transforms,
)
from src.eval import collect_predictions, confusion_matrix, per_class_metrics
from src.train import evaluate, train_one_epoch


class TinyClassifier(nn.Module):
    """Bag-of-pixels logistic regression — small enough to train on CPU in milliseconds."""

    def __init__(self, image_size: int = 32, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * image_size * image_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def synthetic_loader(tmp_path: Path) -> DataLoader:
    rows: list[tuple[str, int, str]] = []
    rng = np.random.default_rng(0)
    for cls_idx, cls_name in enumerate(CLASSES):
        for i in range(8):
            split = "train" if i < 6 else "test"
            d = tmp_path / "images" / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
            p = d / f"{cls_idx:02d}_{i:02d}.png"
            Image.fromarray(arr).save(p)
            rows.append((p.relative_to(tmp_path).as_posix(), cls_idx, split))
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "label", "split"]); w.writerows(rows)

    ds = FER2013Dataset(tmp_path, split="train", transform=build_transforms(train=False, image_size=32))
    return DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)


def test_train_one_epoch_runs(synthetic_loader: DataLoader) -> None:
    model = TinyClassifier(image_size=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(
        model, synthetic_loader, optimizer, scheduler=None, scaler=None,
        device=torch.device("cpu"), label_smoothing=0.1, grad_clip=1.0,
    )
    assert "loss" in metrics and "acc" in metrics
    assert metrics["loss"] > 0


def test_evaluate_runs(synthetic_loader: DataLoader) -> None:
    model = TinyClassifier(image_size=32)
    metrics = evaluate(model, synthetic_loader, torch.device("cpu"))
    assert 0.0 <= metrics["war"] <= 1.0
    assert 0.0 <= metrics["uar"] <= 1.0
    assert metrics["loss"] > 0


def test_confusion_matrix_shape() -> None:
    pred = np.array([0, 1, 2, 0, 1])
    true = np.array([0, 1, 1, 0, 2])
    cm = confusion_matrix(pred, true, NUM_CLASSES)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert cm[0, 0] == 2  # both 0->0
    assert cm[1, 1] == 1  # 1->1
    assert cm[1, 2] == 1  # 1->2 (one mistake)
    assert cm[2, 1] == 1  # 2->1


def test_per_class_metrics_handles_empty_classes() -> None:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm[0, 0] = 5  # 5 correct on class 0
    cm[1, 0] = 2  # class 1 always predicted as 0
    metrics = per_class_metrics(cm)
    assert metrics[0]["recall"] == 1.0  # all class-0 correct
    assert metrics[1]["recall"] == 0.0  # no class-1 correct
    # Empty support classes return 0s, not NaN.
    for m in metrics[2:]:
        assert m["recall"] == 0.0 and m["precision"] == 0.0


def test_collect_predictions_returns_arrays(synthetic_loader: DataLoader) -> None:
    model = TinyClassifier(image_size=32)
    pred, true = collect_predictions(model, synthetic_loader, torch.device("cpu"))
    assert pred.shape == true.shape
    assert pred.dtype.kind == "i"  # integer class indices
    assert (pred >= 0).all() and (pred < NUM_CLASSES).all()
