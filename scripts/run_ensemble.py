"""CLI wrapper around src.ensemble.ensemble_eval.

Usage:
  python scripts/run_ensemble.py \
    --ckpts dan:runs/dan_fer2013/best.pth convnext_tiny:runs/convnext_fer2013/best.pth \
    --dataset fer2013 --root data/fer2013 --tta \
    --out runs/ensemble_dan_convnext/

Each --ckpts entry is `<model_name>:<ckpt_path>`. Model names must be registered
in `src.models.MODEL_REGISTRY` (currently: dan, convnext_tiny, poster_pp).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import CLASSES, NUM_CLASSES
from src.ensemble import ensemble_eval
from src.eval import confusion_matrix, per_class_metrics, plot_confusion_matrix


def parse_ckpt_spec(s: str) -> tuple[str, Path]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--ckpts entry must be 'name:path', got {s!r}")
    name, path = s.split(":", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True, type=parse_ckpt_spec,
                        help="Space-separated model:ckpt entries")
    parser.add_argument("--dataset", choices=["fer2013", "rafdb"], default="fer2013")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Optional per-model averaging weights")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = ensemble_eval(
        ckpt_specs=args.ckpts,
        dataset_name=args.dataset,
        dataset_root=args.root,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta=args.tta,
        weights=args.weights,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pred = np.array(result["predictions"])
    true = np.array(result["labels"])
    cm = confusion_matrix(pred, true, NUM_CLASSES)
    pcm = per_class_metrics(cm)

    metrics = {
        "experiment": args.out.name,
        "ensemble_members": [f"{n}:{p}" for n, p in args.ckpts],
        "dataset": args.dataset,
        "tta": result["tta"],
        "n_samples": result["n_samples"],
        "war": result["war"],
        "uar": result["uar"],
        "per_model_war": result["per_model_war"],
        "weights": result["weights"],
        "per_class": pcm,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(cm, args.out / "confusion_matrix.png")

    print(f"Ensemble WAR={result['war']:.4f}  UAR={result['uar']:.4f}  (TTA={result['tta']})")
    print(f"Per-model WAR: {result['per_model_war']}")
    for m in pcm:
        print(f"  {m['class']:<8s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"Wrote {args.out}/metrics.json and confusion_matrix.png")


if __name__ == "__main__":
    main()
