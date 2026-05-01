"""Prepare RAF-DB from the official release.

Expected layout after EULA-gated download from
http://www.whdeng.cn/RAF/model1.html (report.md:2208):

    data/rafdb/
      Image/aligned/                   # pre-aligned 100x100 RGB face crops
      EmoLabel/list_patition_label.txt # 'train_<n>.jpg <label>' or 'test_<n>.jpg <label>'

The official basic-7 split is 12,271 train / 3,068 test (report.md:2215).
RAF-DB labels in list_patition_label.txt are 1-indexed; we re-map to 0-indexed
to match torchvision conventions.

RAF-DB official label mapping (1..7):
    1 Surprise, 2 Fear, 3 Disgust, 4 Happy, 5 Sad, 6 Angry, 7 Neutral

We ALSO emit the 0-indexed FER-2013-compatible mapping
    0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral
in a second column so cross-dataset evaluation lines up class indices.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# RAF-DB label index (1-indexed in the file) -> emotion name
RAFDB_NAME = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}
# Project-canonical (matches FER-2013 CLASSES order)
CANONICAL_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NAME_TO_CANONICAL = {n: i for i, n in enumerate(CANONICAL_ORDER)}


def main(root: Path) -> None:
    label_file = root / "EmoLabel" / "list_patition_label.txt"
    aligned_dir = root / "Image" / "aligned"
    if not label_file.exists():
        raise FileNotFoundError(
            f"Expected {label_file}. Did you unpack the RAF-DB EULA-gated download here?"
        )
    if not aligned_dir.exists():
        raise FileNotFoundError(
            f"Expected aligned faces at {aligned_dir}. RAF-DB ships pre-aligned 100x100 crops."
        )

    rows: list[tuple[str, int, str]] = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, raf_label = line.split()
            raf_label_int = int(raf_label)  # 1..7
            emotion = RAFDB_NAME[raf_label_int]
            canonical_label = NAME_TO_CANONICAL[emotion]

            stem = Path(name).stem
            aligned_name = f"{stem}_aligned.jpg"
            img_path = aligned_dir / aligned_name
            if not img_path.exists():
                # Some releases keep the original filename; fall back.
                img_path = aligned_dir / name

            split = "train" if name.startswith("train") else "test"
            rel = img_path.relative_to(root).as_posix()
            rows.append((rel, canonical_label, split))

    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r[2] == "train")
    n_test = sum(1 for r in rows if r[2] == "test")
    print(f"Wrote {len(rows)} rows ({n_train} train, {n_test} test) to {manifest}")
    if n_train != 12271 or n_test != 3068:
        print(f"WARNING: expected 12271/3068 train/test (report.md:2215), got {n_train}/{n_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/rafdb"))
    args = parser.parse_args()
    main(args.root)
