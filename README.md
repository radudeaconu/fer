# Facial Emotion Recognition — CV Course Project

A 4-week course project that fine-tunes published FER models on RAF-DB and ships a Gradio demo.

## Datasets and models

| Role | Choice | Justification (from research report) |
|------|--------|--------------------------------------|
| Primary dataset | **RAF-DB** | 29,672 in-the-wild images, 7 classes, every model in the report reports on it (report.md:2186) |
| Backup dataset | **FER-2013** | Free Kaggle download, no EULA, same 7 classes (report.md:1906) |
| Baseline model | **DAN** (ResNet-18) | 89.70% on RAF-DB, public weights, runs on a single T4 (report.md:252) |
| Stretch model | **POSTER++** | 92.21% on RAF-DB SOTA, public weights, dual-stream CNN+Transformer (report.md:905) |

Full SOTA report: `research/facial-emotion-recognition/report.md`.

## Quickstart (Colab)

1. Open `notebooks/01_colab_setup.ipynb` in Google Colab.
2. Run all cells: mounts Drive, clones the DAN and POSTER_V2 reference repos into `third_party/`, installs `requirements.txt`, unpacks the dataset zip from your Drive into `/content/data/`.
3. For training: `notebooks/02_train_dan.ipynb` (Week 2) or `notebooks/03_train_poster.ipynb` (Week 3).
4. For analysis: `notebooks/04_analysis.ipynb` (Week 4).

## Local demo

```bash
pip install -r requirements.txt
python app/gradio_app.py
```
Opens a Gradio interface on `http://localhost:7860/` with image upload + webcam tabs and a model selector.

## Project structure

```
src/         library code (datasets, models, train/eval, inference)
app/         Gradio demo
configs/     YAML hyperparameter configs
notebooks/   Colab entrypoints
scripts/     dataset preparation
tests/       pytest smoke tests
research/    SOTA report (read-only)
```

## Workflow

This repo follows a strict change-log + commit-per-change discipline. See `CLAUDE.md` for the rules and `CHANGES.md` for the rolling log.
