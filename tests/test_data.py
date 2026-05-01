"""Smoke tests for src.data — exercise the full pipeline on a synthetic mini-dataset.

We build a 14-row manifest (2 per class) of small RGB PNGs in a tmp dir,
then verify dataset length, sample shape, label range, and the
class-weighted sampler's distribution.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data import (
    CLASSES,
    NUM_CLASSES,
    FER2013Dataset,
    RAFDBDataset,
    build_transforms,
    class_weighted_sampler,
)


def _make_synthetic_dataset(root: Path, n_per_class: int = 2, image_size: int = 48) -> None:
    """Create a synthetic manifest + images mimicking the FER-2013 layout."""
    rows: list[tuple[str, int, str]] = []
    rng = np.random.default_rng(0)
    for cls_idx, cls_name in enumerate(CLASSES):
        for i in range(n_per_class):
            split = "train" if i < n_per_class - 1 else "test"
            cls_dir = root / "images" / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
            path = cls_dir / f"{cls_idx:02d}_{i:02d}.png"
            Image.fromarray(arr).save(path)
            rel = path.relative_to(root).as_posix()
            rows.append((rel, cls_idx, split))

    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    _make_synthetic_dataset(tmp_path)
    return tmp_path


def test_fer2013_dataset_basic(synthetic_root: Path) -> None:
    ds = FER2013Dataset(synthetic_root, split="train", transform=build_transforms(train=False, image_size=64))
    assert len(ds) == NUM_CLASSES * 1  # 1 train sample per class
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= label < NUM_CLASSES


def test_rafdb_dataset_basic(synthetic_root: Path) -> None:
    ds = RAFDBDataset(synthetic_root, split="train", transform=build_transforms(train=True, image_size=64))
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= label < NUM_CLASSES


def test_dataloader_batching(synthetic_root: Path) -> None:
    ds = FER2013Dataset(synthetic_root, split="train", transform=build_transforms(train=False, image_size=64))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 64, 64)
    assert labels.dtype.is_floating_point is False
    assert labels.min().item() >= 0
    assert labels.max().item() < NUM_CLASSES


def test_class_weighted_sampler_balances_rare_classes() -> None:
    # Heavily imbalanced labels: class 0 dominates, class 6 is rare.
    labels = [0] * 1000 + [1, 2, 3, 4, 5] * 10 + [6]
    sampler = class_weighted_sampler(labels)
    drawn = [labels[i] for i in list(sampler)[:5000]]
    counts = np.bincount(drawn, minlength=NUM_CLASSES)
    # Each class should be sampled within a factor of ~3x of the most-sampled
    # class — far better than the 1000:1 raw imbalance.
    assert counts.max() / counts.min() < 3.0, f"sampler imbalance still high: {counts.tolist()}"


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        FER2013Dataset(tmp_path, split="train")
