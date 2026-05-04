# Colab Cheatsheet — FER 48h Sprint

Exact copy-paste cells for a fresh Colab T4. Assumes:
- Private repo at `github.com/radudeaconu/fer`.
- A fine-grained PAT named `GH_TOKEN` in Colab Secrets (sidebar → key icon).
- `fer2013.csv` (or `.zip`) on Drive at `/MyDrive/fer-data/`.

## How persistence works

Each Colab notebook gets a fresh ephemeral `/content/`. The repo is cloned per session, but **data + run artifacts live on Drive** at `/content/drive/MyDrive/fer-workspace/`:

```
fer-workspace/
├── data_archives/fer2013_prepared.tar.gz   # populated once by 01_colab_setup
└── runs/                                    # all best.pth / metrics.json / cms
```

`scripts/colab_bootstrap.py` hydrates `data/fer2013/` from the archive (fast local SSD — Drive FUSE is too slow to train against) and symlinks `runs/` → Drive so every notebook sees the same checkpoints.

## 0 · Bootstrap (every fresh runtime, every notebook)

This is the same cell at the top of `02_train_dan.ipynb`, `03_train_poster.ipynb`, `04_analysis.ipynb`, `05_train_convnext.ipynb`. If you're starting from a blank notebook, paste:

```python
import os
from pathlib import Path
from google.colab import drive, userdata

drive.mount('/content/drive')
if not Path('/content/fer').exists():
    os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
    !git clone https://$GH_TOKEN@github.com/radudeaconu/fer.git /content/fer
%cd /content/fer
!pip install -q -r requirements.txt
%run scripts/colab_bootstrap.py
```

After this cell, `data/fer2013/` is populated locally and `runs/` is the Drive workspace.

## 1 · One-time data prep (run `01_colab_setup.ipynb` once, ever)

Place `fer2013.csv` (or `fer2013.zip`) in `MyDrive/fer-data/`, then open `notebooks/01_colab_setup.ipynb` and Run All. It clones the repo, runs `prepare_fer2013.py`, then **publishes the prepared data as `fer2013_prepared.tar.gz` to the Drive workspace**. After this you never need to re-run prep — every other notebook hydrates from the tarball in ~10 s.

Verify: `ls /content/drive/MyDrive/fer-workspace/data_archives/` shows `fer2013_prepared.tar.gz` (~30–60 MB).

## 2 · Train DAN (≈50 min)

```python
!python -m src.train --config configs/dan_fer2013.yaml
# checkpoints land in runs/dan_fer2013/{last.pth, best.pth}
```

If Colab disconnects, resume:
```python
!python -m src.train --config configs/dan_fer2013.yaml --resume runs/dan_fer2013/last.pth
```

## 3 · Train ConvNeXt-Tiny (≈65 min)

```python
!python -m src.train --config configs/convnext_fer2013.yaml
# → runs/convnext_fer2013/best.pth
```

## 4 · Single-crop eval (~1 min each)

```python
!python -m src.eval --config configs/dan_fer2013.yaml      --ckpt runs/dan_fer2013/best.pth
!python -m src.eval --config configs/convnext_fer2013.yaml --ckpt runs/convnext_fer2013/best.pth
# metrics.json + confusion_matrix.png → runs/<exp>/eval/
```

## 5 · TTA eval (~3 min each — 10× the forward passes)

```python
!python -m src.eval --config configs/dan_fer2013.yaml      --ckpt runs/dan_fer2013/best.pth      --tta
!python -m src.eval --config configs/convnext_fer2013.yaml --ckpt runs/convnext_fer2013/best.pth --tta
# → runs/<exp>/eval_tta/
```

## 6 · Ensemble (~5 min with TTA)

```python
!python scripts/run_ensemble.py \
  --ckpts dan:runs/dan_fer2013/best.pth convnext_tiny:runs/convnext_fer2013/best.pth \
  --tta \
  --out runs/ensemble_dan_convnext_tta/
# also without TTA for comparison:
!python scripts/run_ensemble.py \
  --ckpts dan:runs/dan_fer2013/best.pth convnext_tiny:runs/convnext_fer2013/best.pth \
  --out runs/ensemble_dan_convnext/
```

## 7 · Ablations (3 × ≈17 min, sequential)

```python
!python -m src.train --config configs/ablations/dan_no_sampler.yaml
!python -m src.train --config configs/ablations/dan_no_augment.yaml
!python -m src.train --config configs/ablations/dan_no_imagenet.yaml

!python -m src.eval --config configs/ablations/dan_no_sampler.yaml  --ckpt runs/dan_no_sampler/best.pth
!python -m src.eval --config configs/ablations/dan_no_augment.yaml  --ckpt runs/dan_no_augment/best.pth
!python -m src.eval --config configs/ablations/dan_no_imagenet.yaml --ckpt runs/dan_no_imagenet/best.pth
```

## 8 · Drive persistence — automatic

`runs/` is a symlink to `/content/drive/MyDrive/fer-workspace/runs/`. Every checkpoint, `metrics.json`, and `confusion_matrix.png` is already on Drive — no manual copy step needed. Run this once per session to verify:

```python
!readlink runs && ls /content/drive/MyDrive/fer-workspace/runs
```

## 9 · Run analysis notebook

Open `notebooks/04_analysis.ipynb` in Colab → Run All. Expect:
- A 6-row comparison table.
- 1×2 confusion-matrix subplot.
- Per-class recall delta bars.
- Failure gallery per model.
- Agreement counts + oracle upper bound.

Save the figure outputs:
```python
!mkdir -p figures
# In each plotting cell, add:  plt.savefig('figures/<name>.png', dpi=150, bbox_inches='tight')
```

## 10 · Commit figures + push

```python
!git add figures/
!git commit -m "docs(figures): add eval figures for course report"
!git push origin master
```

## 11 · Gradio screenshots (local, not Colab)

On your laptop:
```bash
# pull the trained checkpoints from Drive into runs/ first
python -m app.gradio_app
# → http://127.0.0.1:7860, Compare tab, screenshot DAN/ConvNeXt disagreements
```

---

## Common pitfalls

- **`ModuleNotFoundError: src`** — you're not in `/content/fer`. Re-run §0.
- **OOM on ConvNeXt** — drop batch to 32 in the config.
- **TTA gives identical numbers to no-TTA** — the TenCrop transform isn't being applied; check that `--tta` is actually present in the command.
- **Disconnect mid-train** — re-run §0 in the new runtime; resume with `--resume runs/<exp>/last.pth`.
- **`gh: command not found` on Colab** — you don't need gh on Colab; the bootstrap clones via `GH_TOKEN`.
- **`[bootstrap] WARNING: no archive at .../fer2013_prepared.tar.gz`** — you haven't run `01_colab_setup.ipynb` yet. Open it and Run All; subsequent notebooks will hydrate automatically.
- **`runs/` is empty in `04_analysis.ipynb` even though I trained earlier** — confirm `readlink runs` points to `/content/drive/MyDrive/fer-workspace/runs/`. If not, the bootstrap was skipped — re-run §0.

## Time budget reality check

| Block | Wall time |
|-------|-----------|
| Data prep | 5 min |
| DAN train | 50 min |
| ConvNeXt train | 65 min |
| Eval × 2 (single-crop + TTA) | 8 min |
| Ensemble × 2 | 8 min |
| Ablations × 3 | 50 min |
| Analysis notebook | 15 min |
| Drive sync + push | 10 min |
| **Total active compute** | **~3.5 h** |

Colab free tier ≈ 4–6 h before hitting limits — fits, but kick off DAN training **first** because everything depends on it.
