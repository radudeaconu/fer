"""Evaluation: WAR, UAR, per-class precision/recall/F1, confusion matrix.

Run: python -m src.eval --config configs/dan_rafdb.yaml --ckpt runs/dan_rafdb/best.pth

Outputs (under runs/<experiment_name>/eval/):
    metrics.json          {war, uar, per_class: [{class, precision, recall, f1, support}], ...}
    confusion_matrix.png  normalized + raw counts
    confusion_matrix.csv  raw counts for downstream analysis
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (
    CLASSES,
    NUM_CLASSES,
    FER2013Dataset,
    RAFDBDataset,
    build_transforms,
)
from src.models import build_model

DATASETS = {"fer2013": FER2013Dataset, "rafdb": RAFDBDataset}


def _logits(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    raise TypeError(f"Cannot extract logits from {type(out)}")


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = _logits(model(imgs))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        trues.extend(labels.tolist())
    return np.array(preds), np.array(trues)


def confusion_matrix(pred: np.ndarray, true: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> list[dict]:
    out = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        pred_count = int(cm[:, c].sum())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append({
            "class": CLASSES[c],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    return out


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(cfg_path: Path, ckpt: Path) -> None:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Cls = DATASETS[cfg["data"]["dataset"]]
    eval_split = "test"
    test_ds = Cls(
        cfg["data"]["root"],
        split=eval_split,
        transform=build_transforms(train=False, image_size=cfg["data"]["image_size"]),
    )
    loader = DataLoader(test_ds, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["data"]["num_workers"])

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model", state.get("model_state_dict", state))
    model.load_state_dict(sd, strict=False)

    pred, true = collect_predictions(model, loader, device)
    cm = confusion_matrix(pred, true, NUM_CLASSES)
    war = float((pred == true).mean())
    per_class = per_class_metrics(cm)
    uar = float(np.mean([m["recall"] for m in per_class]))

    out_dir = Path("runs") / cfg["experiment_name"] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": cfg["experiment_name"],
        "dataset": cfg["data"]["dataset"],
        "split": eval_split,
        "n_samples": int(len(true)),
        "war": round(war, 4),
        "uar": round(uar, 4),
        "per_class": per_class,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    with (out_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *CLASSES])
        for i, row in enumerate(cm):
            w.writerow([CLASSES[i], *row.tolist()])

    print(f"WAR={war:.4f}  UAR={uar:.4f}")
    for m in per_class:
        print(f"  {m['class']:<8s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"Wrote {out_dir}/metrics.json and confusion_matrix.{{png,csv}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.ckpt)
