# Colab Cheatsheet — FER 48h Sprint

Exact copy-paste cells for a fresh Colab T4. Assumes the repo is private at `github.com/radudeaconu/fer` and FER-2013 (`fer2013.csv` or `fer2013.zip`) is on Google Drive at `/MyDrive/datasets/`.

## 0 · Mount Drive + clone repo (every fresh runtime)

```python
from google.colab import drive
drive.mount('/content/drive')

# gh PAT lives in Colab "Secrets" as GH_TOKEN — Settings → Secrets → enable for this notebook
from google.colab import userdata
import os, subprocess
os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
!git clone https://$GH_TOKEN@github.com/radudeaconu/fer.git
%cd /content/fer
!pip install -q -r requirements.txt
```

## 1 · Data setup (once per runtime)

```python
# Run notebooks/01_colab_setup.ipynb cells, OR inline:
!mkdir -p data
!cp /content/drive/MyDrive/datasets/fer2013.csv data/   # or fer2013.zip
!python scripts/prepare_fer2013.py --csv data/fer2013.csv --out data/fer2013/
```

Verify: `data/fer2013/{train,val,test}/<class>/*.png` exists; ~28k train, ~3.5k val, ~3.6k PrivateTest.

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

## 8 · Persist artifacts back to Drive (do this often)

```python
import shutil, os
os.makedirs('/content/drive/MyDrive/fer_runs', exist_ok=True)
for exp in ['dan_fer2013', 'convnext_fer2013',
            'dan_no_sampler', 'dan_no_augment', 'dan_no_imagenet',
            'ensemble_dan_convnext', 'ensemble_dan_convnext_tta']:
    src = f'runs/{exp}'
    if os.path.isdir(src):
        shutil.copytree(src, f'/content/drive/MyDrive/fer_runs/{exp}', dirs_exist_ok=True)
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

- **`ModuleNotFoundError: src`** — you're not in `/content/fer`. `cd` first.
- **OOM on ConvNeXt** — drop batch to 32 in the config.
- **TTA gives identical numbers to no-TTA** — the TenCrop transform isn't being applied; check that `--tta` is actually present in the command.
- **Disconnect mid-train** — re-mount Drive; the resume flag picks up from `last.pth`.
- **`gh: command not found` on Colab** — you don't need gh on Colab; use `git clone https://$GH_TOKEN@github.com/...` as in §0.

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
