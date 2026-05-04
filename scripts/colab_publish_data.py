"""Tar up data/fer2013/ and stage it on Drive for cross-notebook reuse.

Run after notebooks/01_colab_setup.ipynb finishes prepare_fer2013.py.
Idempotent: re-running with --force replaces the existing archive.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/fer")
WS = Path("/content/drive/MyDrive/fer-workspace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing archive")
    args = ap.parse_args()

    src = REPO / "data" / "fer2013"
    if not src.exists() or not any(src.iterdir()):
        sys.exit(f"[publish] {src} is missing or empty; run prepare_fer2013.py first.")

    (WS / "data_archives").mkdir(parents=True, exist_ok=True)
    archive = WS / "data_archives" / "fer2013_prepared.tar.gz"
    if archive.exists() and not args.force:
        print(f"[publish] {archive} already exists ({archive.stat().st_size / 1e6:.1f} MB); use --force to replace.")
        return
    if archive.exists():
        archive.unlink()
    print(f"[publish] tarring {src} -> {archive} (this takes a few minutes for 36k PNGs)")
    subprocess.check_call(["tar", "-czf", str(archive), "-C", str(REPO / "data"), "fer2013"])
    print(f"[publish] done. archive size: {archive.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
