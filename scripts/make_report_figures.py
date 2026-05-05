"""Build comparison figures for the course report from metrics JSONs.

Runs locally (no GPU, no dataset access) — reads:
    runs/<exp>/eval/metrics.json
    runs/<exp>/eval/confusion_matrix.csv
    runs/ensemble*/metrics.json

Writes:
    figures/cm_dan_vs_convnext.png    -- 1x2 row-normalized CMs, shared colorbar
    figures/perclass_recall_delta.png -- ConvNeXt - DAN recall per emotion
    figures/headline_war_uar.png      -- grouped bars for the 6-row comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_cm(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def cm_subplot() -> Path:
    dan = load_cm(ROOT / "runs/dan_fer2013/eval/confusion_matrix.csv")
    cnx = load_cm(ROOT / "runs/convnext_fer2013/eval/confusion_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, cm, title in [(axes[0], dan, "DAN"), (axes[1], cnx, "ConvNeXt-Tiny")]:
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(7), CLASSES)
        war = np.trace(cm) / cm.sum()
        ax.set_title(f"{title}  (WAR={war:.3f})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if title == "DAN" else "")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes, shrink=0.85, label="row-normalized rate")
    out = FIG / "cm_dan_vs_convnext.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def perclass_delta() -> Path:
    dan = json.loads((ROOT / "runs/dan_fer2013/eval/metrics.json").read_text())["per_class"]
    cnx = json.loads((ROOT / "runs/convnext_fer2013/eval/metrics.json").read_text())["per_class"]
    by_cls = {p["class"]: p for p in dan}
    delta = [next(p for p in cnx if p["class"] == c)["recall"] - by_cls[c]["recall"] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
    ax.bar(CLASSES, delta, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("recall(ConvNeXt-Tiny) − recall(DAN)")
    ax.set_title("Per-class recall delta on FER-2013 PrivateTest\n(green = ConvNeXt wins, red = DAN wins)")
    for i, d in enumerate(delta):
        ax.text(i, d + (0.005 if d >= 0 else -0.012), f"{d:+.3f}", ha="center", fontsize=9)
    ax.set_ylim(min(delta) - 0.04, max(delta) + 0.04)
    out = FIG / "perclass_recall_delta.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def headline_bars() -> Path:
    rows = []
    for label, path in [
        ("DAN", "runs/dan_fer2013/eval/metrics.json"),
        ("ConvNeXt-Tiny", "runs/convnext_fer2013/eval/metrics.json"),
        ("Ensemble", "runs/ensemble_dan_convnext/metrics.json"),
        ("Ensemble + TTA", "runs/ensemble_dan_convnext_tta/metrics.json"),
    ]:
        d = json.loads((ROOT / path).read_text())
        rows.append((label, d["war"], d["uar"]))
    # Add per-model TTA from ensemble_tta.per_model_war (matches a standalone --tta eval)
    tta = json.loads((ROOT / "runs/ensemble_dan_convnext_tta/metrics.json").read_text())
    rows.insert(2, ("ConvNeXt-Tiny + TTA", tta["per_model_war"][1], None))
    rows.insert(2, ("DAN + TTA", tta["per_model_war"][0], None))

    labels = [r[0] for r in rows]
    war = [r[1] for r in rows]
    uar = [r[2] if r[2] is not None else np.nan for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bw = 0.4
    ax.bar(x - bw / 2, war, bw, label="WAR", color="#1f77b4")
    ax.bar(x + bw / 2, uar, bw, label="UAR", color="#ff7f0e")
    ax.axhline(0.665, linestyle="--", color="gray", linewidth=1)
    ax.text(len(labels) - 0.5, 0.668, "human ≈ 0.665", color="gray", fontsize=8, ha="right")
    for i, w in enumerate(war):
        ax.text(i - bw / 2, w + 0.005, f"{w:.3f}", ha="center", fontsize=8)
    for i, u in enumerate(uar):
        if not np.isnan(u):
            ax.text(i + bw / 2, u + 0.005, f"{u:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0.55, 0.80)
    ax.set_ylabel("Accuracy on FER-2013 PrivateTest (n=3589)")
    ax.set_title("Headline comparison: DAN vs ConvNeXt-Tiny vs Ensemble (with/without TTA)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIG / "headline_war_uar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    for fn in [cm_subplot, perclass_delta, headline_bars]:
        path = fn()
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    sys.exit(main())
