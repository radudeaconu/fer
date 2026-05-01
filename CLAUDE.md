# CLAUDE.md — Project Rules for FER

This file governs how Claude Code works in this repository. Read it at the start of every session.

## Project context

A 4-week computer-vision course project on facial emotion recognition (FER). The authoritative reference is `research/facial-emotion-recognition/report.md` (26 models + 24 datasets, generated 2026-04-30). Cite specific lines from the report when justifying choices.

**Decisions (locked):**
- Primary dataset: **RAF-DB** — 29,672 in-the-wild static images, 7 classes, EULA required (report.md:2186).
- Backup dataset: **FER-2013** — 35,887 images, free Kaggle download (report.md:1906).
- Baseline model: **DAN** (ResNet-18, 89.70% RAF-DB) — https://github.com/yaoing/DAN (report.md:252).
- Stretch model: **POSTER++** (43.7M params, 92.21% RAF-DB) — https://github.com/Talented-Q/POSTER_V2 (report.md:905).
- Hardware: Google Colab / Kaggle free tier (T4 GPU). Code must run on a single T4 with ≤16 GB RAM.
- Strategy: **fine-tune from released checkpoints only** — never train DAN or POSTER++ from random init.
- Demo: Gradio web app.

## Workflow rules (enforced)

**Rule 1 — Document every change in `CHANGES.md`.**
After every code edit (every file write or edit), append an entry to `CHANGES.md` under today's `## YYYY-MM-DD` heading in this format:
```
- <relative/path/to/file>: <what changed in one line> — <why>
```
The "why" is non-negotiable. If the why is obvious from the file diff, the entry is too thin — explain the motivation, not the diff.

**Rule 2 — One logical change = one git commit.**
After every logically-complete change, run:
```
git add <specific files>
git commit -m "<type(scope): concise message>"
```
Conventional types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`. Never use `git add -A` or `git add .` — stage files explicitly to avoid accidentally committing data, checkpoints, or local notebooks. Never use `--no-verify`.

**Rule 3 — `research/` is read-only.**
Never modify any file under `research/`. It is the authoritative reference; treat it as immutable. Read it for guidance; cite line numbers when justifying decisions.

**Rule 4 — Never commit data, checkpoints, or training runs.**
`.gitignore` excludes `data/`, `checkpoints/`, `runs/`, `third_party/`, `*.pth`, `*.tar`, `*.zip`. Don't add exceptions. Eval figures (PNGs of confusion matrices, attention overlays) under `runs/<exp>/eval/` may be committed if small (<1 MB) and useful for the course report — copy them to `figures/` and commit from there.

**Rule 5 — Match metric definitions to the report.**
- WAR (Weighted Accuracy / Overall Accuracy) and UAR (Unweighted / Mean-class Accuracy) must both be reported, because RAF-DB and AffectNet are imbalanced (report.md:2210, 1371).
- Always report on the official RAF-DB test split (3,068 images) and FER-2013 PrivateTest split (3,589 images).
- For POSTER++ specifically: never train from random init — always start from IR-50 + MobileFaceNet released backbones (report.md:933–934).

## Repository layout

```
src/                 # Library code (data.py, models.py, train.py, eval.py, predict.py)
app/                 # Gradio demo
configs/             # YAML hyperparameter configs (one per experiment)
notebooks/           # Colab entrypoints (numbered: 01_, 02_, ...)
scripts/             # One-shot data prep
tests/               # pytest smoke tests
third_party/         # gitignored — cloned DAN + POSTER_V2 reference repos
research/            # READ-ONLY reference report and structured JSON
runs/                # gitignored — training output (logs, ckpts, eval/)
data/                # gitignored — datasets
figures/             # eval figures worth committing for the course report
```

## Coding conventions

- Python 3.10+, PyTorch ≥2.0.
- One responsibility per file in `src/`.
- Configs are YAML, loaded once at the top of `train.py` / `eval.py`.
- Log to `runs/<exp_name>/` (TensorBoard or plain JSON — pick one and stick with it).
- No silent fallbacks: if a checkpoint or dataset path is missing, raise with a clear message including the path that was tried.

## Plan reference

The full build plan lives at `C:\Users\Lenovo\.claude\plans\use-the-report-to-parsed-raccoon.md`. Re-read it before starting a new phase.
