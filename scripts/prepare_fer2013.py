"""Prepare FER-2013 from the Kaggle CSV release.

Input:  data/fer2013/fer2013.csv  (downloaded from
        https://www.kaggle.com/datasets/msambare/fer2013)
Output: data/fer2013/images/<split>/<class>/<idx>.png
        data/fer2013/manifest.csv  (path,label,split)

Splits follow the original Goodfellow et al. 2013 partitioning
(report.md:1934): Training / PublicTest / PrivateTest.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def main(csv_path: Path, out_root: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected FER-2013 CSV at {csv_path}. "
            "Download from https://www.kaggle.com/datasets/msambare/fer2013"
        )

    images_root = out_root / "images"
    manifest_rows: list[tuple[str, int, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(tqdm(reader, desc="fer2013")):
            label = int(row["emotion"])
            usage = row["Usage"]
            split = USAGE_TO_SPLIT[usage]
            pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)

            cls_dir = images_root / split / CLASSES[label]
            cls_dir.mkdir(parents=True, exist_ok=True)
            img_path = cls_dir / f"{idx:06d}.png"
            Image.fromarray(pixels, mode="L").save(img_path)

            manifest_rows.append(
                (str(img_path.relative_to(out_root)).replace("\\", "/"), label, split)
            )

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/fer2013/fer2013.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/fer2013"))
    args = parser.parse_args()
    main(args.csv, args.out)
