"""Colab workspace bootstrap. Run from the cloned repo (/content/fer) after pip install.

Idempotent — safe to re-run within a session.

Drive layout this script maintains:

    /content/drive/MyDrive/fer-workspace/
        data_archives/fer2013_prepared.tar.gz   # produced by 01_colab_setup
        runs/                                    # all training + eval outputs

Per-session effect:
    1. Hydrate data/fer2013/ from the tarball on Drive into local /content/fer/data/
       (Drive FUSE is too slow to train against 36k PNGs directly.)
    2. Symlink /content/fer/runs -> Drive workspace runs/ so checkpoints,
       eval JSONs, and confusion-matrix PNGs persist across runtime restarts
       and are visible to every notebook.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/fer")
WS = Path("/content/drive/MyDrive/fer-workspace")


def main() -> None:
    if not REPO.exists():
        sys.exit(f"[bootstrap] expected repo at {REPO}; clone it first.")
    if not Path("/content/drive/MyDrive").exists():
        sys.exit(
            "[bootstrap] Drive is not mounted. Run "
            "`from google.colab import drive; drive.mount('/content/drive')` first."
        )
    WS.mkdir(parents=True, exist_ok=True)
    (WS / "data_archives").mkdir(exist_ok=True)
    (WS / "runs").mkdir(exist_ok=True)

    _hydrate_data()
    _link_runs()
    print(f"[bootstrap] workspace ready at {WS}")


def _hydrate_data() -> None:
    local = REPO / "data" / "fer2013"
    archive = WS / "data_archives" / "fer2013_prepared.tar.gz"
    if local.exists() and any(local.iterdir()):
        print(f"[bootstrap] data/fer2013 already populated ({_count(local)} files); skip hydrate.")
        return
    if not archive.exists():
        print(
            f"[bootstrap] WARNING: no archive at {archive}.\n"
            "             Run notebooks/01_colab_setup.ipynb once to prepare and publish the data."
        )
        return
    print(f"[bootstrap] hydrating data/fer2013 from {archive} ({archive.stat().st_size / 1e6:.1f} MB) ...")
    (REPO / "data").mkdir(exist_ok=True)
    subprocess.check_call(["tar", "-xzf", str(archive), "-C", str(REPO / "data")])
    print(f"[bootstrap] hydrated {_count(local)} files into {local}")


def _link_runs() -> None:
    runs = REPO / "runs"
    target = WS / "runs"
    if runs.is_symlink():
        if Path(os.readlink(runs)) == target:
            print(f"[bootstrap] runs/ already linked to {target}")
            return
        runs.unlink()
    elif runs.exists():
        if any(runs.iterdir()):
            sys.exit(
                f"[bootstrap] {runs} already exists and is non-empty; refusing to overwrite. "
                f"Move its contents to {target} manually, then re-run."
            )
        runs.rmdir()
    runs.symlink_to(target)
    print(f"[bootstrap] runs/ -> {target}")


def _count(p: Path) -> int:
    return sum(1 for _ in p.rglob("*") if _.is_file())


if __name__ == "__main__":
    main()
