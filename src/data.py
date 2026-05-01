"""Dataset classes and transforms for FER-2013 and RAF-DB.

Both datasets use a unified 7-class label space ordered as:
    0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral

This matches the FER-2013 default ordering. The RAF-DB prep script
remaps RAF's native 1..7 ordering to this canonical layout so cross-dataset
evaluation lines up without bookkeeping at training time.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(CLASSES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_manifest(manifest_path: Path, split: str) -> list[tuple[str, int]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            f"Run scripts/prepare_{'fer2013' if 'fer2013' in str(manifest_path) else 'rafdb'}.py first."
        )
    rows: list[tuple[str, int]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == split:
                rows.append((r["path"], int(r["label"])))
    if not rows:
        raise ValueError(f"No rows for split={split!r} in {manifest_path}")
    return rows


class _ManifestDataset(Dataset):
    """Manifest-driven image classification dataset.

    All images are converted to RGB on read (FER-2013 source is grayscale and
    is tiled to 3 channels so ImageNet-pretrained backbones — DAN's ResNet-18,
    POSTER++'s IR-50 — accept the input without further channel handling).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = _read_manifest(self.root / "manifest.csv", split)
        self.labels = [lbl for _, lbl in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        with Image.open(self.root / rel_path) as im:
            img = im.convert("RGB").copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class FER2013Dataset(_ManifestDataset):
    """FER-2013 (48x48 grayscale, in-the-wild). report.md:1906.

    Splits: 'train' (Training, ~28,709) | 'val' (PublicTest, 3,589) | 'test' (PrivateTest, 3,589).
    """


class RAFDBDataset(_ManifestDataset):
    """RAF-DB basic-7 (100x100 aligned RGB). report.md:2186.

    Splits: 'train' (12,271) | 'test' (3,068). No official val split.
    """


def build_transforms(train: bool, image_size: int = 224) -> Callable:
    """Standard FER preprocessing pipeline.

    Train: RandomResizedCrop, HorizontalFlip, RandAugment(light), ColorJitter, Normalize.
    Eval:  Resize -> CenterCrop -> Normalize.
    Normalization uses ImageNet statistics to match the MS-Celeb-1M-pretrained
    backbones used by DAN and POSTER++ (report.md:278, 933).
    """
    if train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def class_weighted_sampler(labels: list[int], num_classes: int = NUM_CLASSES) -> WeightedRandomSampler:
    """WeightedRandomSampler that inverts class frequency.

    RAF-DB and FER-2013 are both severely imbalanced (Disgust/Fear are rare;
    report.md:2210, 1928). Without inverse-frequency weighting, models trivially
    over-predict Happy/Neutral and inflate WAR while UAR collapses.
    """
    counts = Counter(labels)
    missing = [c for c in range(num_classes) if counts[c] == 0]
    if missing:
        raise ValueError(f"Empty classes in labels: {missing}")
    class_weights = np.array([1.0 / counts[c] for c in range(num_classes)], dtype=np.float64)
    sample_weights = np.array([class_weights[lbl] for lbl in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )
