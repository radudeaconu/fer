# Changes

Rolling log of every code modification. Format: `- <file>: <what> — <why>` under a daily heading.
See `CLAUDE.md` Rule 1 for the contract.

## 2026-04-30

- `.gitignore`: Initial ignore rules — exclude data/, checkpoints/, runs/, third_party/, Python bytecode, notebook checkpoints, and `.claude/` (local Claude tooling state). Keeps the repo focused on source code, not artifacts or per-user tooling.
- `CLAUDE.md`: Authored the project rules file — locks in datasets/models, enforces CHANGES.md + per-change git commits, marks research/ read-only. Required so future sessions stay consistent with the approved plan.
- `CHANGES.md`: Created the change log with seed entries for the bootstrap files. Establishes the documentation discipline from the first commit.
- `requirements.txt`: Pinned floor versions for torch/torchvision/numpy/pandas/pillow/scikit-learn/matplotlib/seaborn/pyyaml/tqdm/gradio/opencv-headless/grad-cam/pytest. Floors (not exact pins) so Colab's preinstalled torch is reused without conflict.
- `scripts/prepare_fer2013.py`: One-shot script to materialize FER-2013 from the Kaggle CSV into per-split per-class PNG folders + manifest.csv. Needed because `src/data.py` is manifest-driven and the Kaggle release ships pixels packed into a single CSV column.
- `scripts/prepare_rafdb.py`: One-shot script to parse `list_patition_label.txt` and remap RAF-DB's native 1..7 labels to the project-canonical 7-class order (matching FER-2013) so cross-dataset evaluation lines up class indices without runtime bookkeeping.
- `src/__init__.py`: Empty package marker so `from src.data import ...` resolves cleanly.
- `src/data.py`: Implemented `FER2013Dataset`, `RAFDBDataset` (both manifest-driven, RGB-tiled), `build_transforms` (train uses RandAugment+ColorJitter+RandomErasing; eval uses Resize+CenterCrop), and `class_weighted_sampler` (inverse-frequency `WeightedRandomSampler` to counter the report-flagged Disgust/Fear imbalance on both datasets).
- `tests/__init__.py`, `tests/test_data.py`: Pytest smoke suite using a synthetic 14-image manifest in tmp_path. Covers shape, label range, DataLoader batching, sampler rebalancing, and the missing-manifest error path. **Note:** local Python 3.13 has no torch installed; tests are intended to run on Colab/Kaggle where torch is preinstalled. Local-laptop testing would require setting up a venv with CPU torch first.
