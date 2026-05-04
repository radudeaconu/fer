"""Multi-model ensemble: average softmax probabilities from independent runs.

Use case: combine DAN (CNN+attention) and ConvNeXt-Tiny (modern CNN) on the same
test set. Models that make different mistakes (low prediction agreement) ensemble
better than models that fail on the same samples.

Importing this module is cheap — no checkpoints loaded until `ensemble_eval` runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval import _logits, build_tta_transform
from src.data import (
    NUM_CLASSES,
    FER2013Dataset,
    RAFDBDataset,
    build_transforms,
)
from src.models import build_model

DATASETS = {"fer2013": FER2013Dataset, "rafdb": RAFDBDataset}


@torch.no_grad()
def _model_softmax_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single model over the loader; return (probs[N, C], labels[N])."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    for imgs, labels in loader:
        if tta:
            b, k, c, h, w = imgs.shape
            imgs = imgs.to(device, non_blocking=True).view(b * k, c, h, w)
            logits = _logits(model(imgs))
            probs = F.softmax(logits, dim=1).view(b, k, -1).mean(dim=1)
        else:
            imgs = imgs.to(device, non_blocking=True)
            logits = _logits(model(imgs))
            probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def ensemble_eval(
    ckpt_specs: list[tuple[str, str | Path]],
    dataset_name: str,
    dataset_root: str | Path,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    tta: bool = False,
    weights: list[float] | None = None,
) -> dict:
    """Evaluate a logits-averaging ensemble.

    Args:
        ckpt_specs: list of (model_name, ckpt_path) tuples — e.g.
            [("dan", "runs/dan_fer2013/best.pth"), ("convnext_tiny", "runs/convnext_fer2013/best.pth")]
        dataset_name: 'fer2013' or 'rafdb'.
        dataset_root: path to the dataset (containing manifest.csv).
        device: torch device.
        image_size: input resolution.
        batch_size: eval batch size (auto-divided by 10 if tta=True).
        tta: whether to use 10-crop TTA per model.
        weights: optional per-model weights for the average. Defaults to uniform.

    Returns:
        dict with keys: war, uar, n_models, n_samples, predictions (list[int]),
        labels (list[int]), per_model_war (list[float]).
    """
    Cls = DATASETS[dataset_name]
    if tta:
        transform = build_tta_transform(image_size=image_size)
        loader_bs = max(1, batch_size // 10)
    else:
        transform = build_transforms(train=False, image_size=image_size)
        loader_bs = batch_size

    test_ds = Cls(dataset_root, split="test", transform=transform)
    loader = DataLoader(test_ds, batch_size=loader_bs, num_workers=num_workers)

    if weights is None:
        weights = [1.0] * len(ckpt_specs)
    if len(weights) != len(ckpt_specs):
        raise ValueError(f"weights length {len(weights)} != ckpt_specs length {len(ckpt_specs)}")
    weight_sum = sum(weights)

    summed: np.ndarray | None = None
    labels_ref: np.ndarray | None = None
    per_model_war: list[float] = []

    for w, (model_name, ckpt_path) in zip(weights, ckpt_specs):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
        model = build_model(model_name).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = state.get("model", state.get("model_state_dict", state.get("state_dict", state)))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

        probs, labels = _model_softmax_probs(model, loader, device, tta=tta)
        per_model_war.append(float((probs.argmax(axis=1) == labels).mean()))

        contribution = (w / weight_sum) * probs
        summed = contribution if summed is None else summed + contribution
        labels_ref = labels  # same loader → same order; assign once

        # Free GPU memory before loading the next model.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert summed is not None and labels_ref is not None
    pred = summed.argmax(axis=1)
    war = float((pred == labels_ref).mean())

    # UAR = mean per-class recall.
    per_class_recall = []
    for c in range(NUM_CLASSES):
        mask = labels_ref == c
        if mask.sum() > 0:
            per_class_recall.append(float((pred[mask] == c).mean()))
    uar = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    return {
        "war": round(war, 4),
        "uar": round(uar, 4),
        "n_models": len(ckpt_specs),
        "n_samples": int(len(labels_ref)),
        "tta": tta,
        "weights": list(weights),
        "per_model_war": [round(w, 4) for w in per_model_war],
        "predictions": pred.tolist(),
        "labels": labels_ref.tolist(),
    }
