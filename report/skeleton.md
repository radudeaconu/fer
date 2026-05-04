# Facial Emotion Recognition: A Comparison Study on FER-2013

**Course:** Computer Vision · **Author:** Radu Deaconu · **Date:** 2026-05-XX
**Repository:** https://github.com/radudeaconu/fer
**Target length:** 5–7 pages + figures.

---

## 1. Introduction (~0.5 pg)

- Problem statement: 7-class facial emotion classification on in-the-wild images.
- Why it matters: HCI, accessibility, affective computing.
- What this report contributes:
  1. A controlled comparison of a **CNN+attention** baseline (DAN) vs. a **modern CNN** (ConvNeXt-Tiny) on FER-2013.
  2. An ablation of three DAN design choices (class-weighted sampling, RandAugment, ImageNet init).
  3. A test-time augmentation (TTA) and 2-model ensemble study.
- Hardware constraint: single T4 GPU, ≤16 GB. Strategy: fine-tune from released ImageNet weights, never train from scratch.
- Hook to literature: cite `research/.../report.md:252` (DAN), `:488` (EmoNeXt/ConvNeXt family), `:1928` (FER-2013 human upper bound ~65–68%).

## 2. Dataset (~0.5 pg)

- **FER-2013**: 35,887 grayscale 48×48 images; PrivateTest split = 3,589 (cite `report.md:1906`).
- 7 classes: angry, disgust, fear, happy, sad, surprise, neutral. Class imbalance ratio ~13× (max:min).
- Pre-processing: resize 48 → 224, replicate to 3 channels, ImageNet normalization.
- **Why FER-2013, not RAF-DB:** RAF-DB requires a EULA that arrived too late for the sprint window; FER-2013 is the documented backup (`CLAUDE.md`).
- Figure: 7-row class-distribution bar chart (train vs. PrivateTest).

## 3. Method (~1.5 pg)

### 3.1 Models
- **DAN** (Distract-Your-Attention Network, 2021): ResNet-18 backbone + multi-head cross-attention + affinity loss. 19.7M params. Cite `report.md:252`.
- **ConvNeXt-Tiny** (2022): pure-convolutional, modernized ResNet design (depth-wise conv, LayerNorm, GELU, 7×7 kernels). 27.8M params. Cite `report.md:488`.
- Both initialized from ImageNet weights; final classifier replaced with `Linear(d → 7)`.

### 3.2 Training
- Optimizer: AdamW. DAN: lr=3e-4, wd=1e-4. ConvNeXt: lr=1e-4, wd=5e-2 (paper default; ImageNet head is more delicate).
- Schedule: linear warmup (2 ep) → cosine decay. 30 epochs.
- Loss: cross-entropy. DAN adds its native affinity + partition losses internally.
- Class imbalance: weighted random sampler, weights ∝ 1/freq.
- Augmentation: RandAugment(N=2, M=9) + horizontal flip + random erasing.
- AMP fp16, batch=64, seed=42.

### 3.3 Test-Time Augmentation
- 10-crop = 5 spatial crops × {original, h-flip}, softmax-averaged.

### 3.4 Ensemble
- Logits averaging of DAN + ConvNeXt-Tiny softmax outputs (uniform weights).

### 3.5 Ablations (DAN, 10 ep each)
| # | Removed | Tests |
|---|---------|-------|
| 1 | class-weighted sampler | Does the 13× imbalance actually need balancing? |
| 2 | RandAugment + h-flip + erase | How much does aug move the needle? |
| 3 | ImageNet init (random init) | Transfer-learning contribution |

### 3.6 Metrics
- **WAR** (Weighted / Overall Accuracy) and **UAR** (Unweighted / mean per-class). Both required because of the imbalance — cite `report.md:2210, 1371`. Confusion matrices normalized by row.

## 4. Results (~2 pg)

### 4.1 Main comparison table

| Model | Params | WAR | UAR | Train time |
|-------|--------|-----|-----|------------|
| DAN, 30 ep | 19.7M | __ | __ | __ min |
| DAN + TTA | — | __ | __ | — |
| ConvNeXt-Tiny, 30 ep | 27.8M | __ | __ | __ min |
| ConvNeXt-Tiny + TTA | — | __ | __ | — |
| Ensemble (DAN + ConvNeXt) | 47.5M | __ | __ | — |
| Ensemble + TTA | 47.5M | __ | __ | — |
| Human (literature) | — | ~65–68 | — | — |

Numbers come from `runs/<exp>/eval[_tta]/metrics.json` and `runs/ensemble*/metrics.json`.

### 4.2 Ablations (DAN, 10 ep)

| Setup | WAR | UAR | Δ vs full |
|-------|-----|-----|-----------|
| DAN full (10 ep) | __ | __ | 0 |
| − class-weighted sampler | __ | __ | __ |
| − RandAugment | __ | __ | __ |
| − ImageNet init | __ | __ | __ |

**Discussion:** which knob carries the most weight? Hypothesis: ImageNet init > augmentation > sampler.

### 4.3 Confusion matrices (1×2 subplot)
DAN vs ConvNeXt-Tiny, both row-normalized, shared colorbar. Drop in `figures/cm_dan_vs_convnext.png`. Notebook cell B.

### 4.4 Per-class recall deltas
Bar chart `ConvNeXt_recall − DAN_recall` per class. Highlights complementary strengths. `figures/perclass_delta.png`. Notebook cell C.

### 4.5 Agreement and oracle upper bound
- % both correct / only DAN / only ConvNeXt / both wrong.
- Oracle ensemble = 1 − (both-wrong rate). Numeric upper bound for any combination of these two models. Notebook cell E.

### 4.6 Failure gallery
Top-10 highest-confidence wrong predictions per model. Diagnose systematic errors (e.g. fear↔surprise confusion). Notebook cell D.

## 5. Discussion (~1 pg)

- Did the modern CNN beat the CNN+attention baseline? By how much, and on which classes?
- Where does TTA help / not help? (Usually +1–2 WAR; if smaller, why?)
- Does the ensemble beat the best individual? If not, the two models are too correlated — discuss honestly. Cross-reference oracle upper bound from §4.5.
- Compare ablation deltas to literature priors.
- Threats to validity: single seed, single split, FER-2013's known label noise.

## 6. Demo (~0.25 pg)
- Gradio app (`app/gradio_app.py`) with three tabs: Image, Webcam, Compare. Two screenshots:
  1. Compare tab on a "tricky" face where DAN and ConvNeXt disagree.
  2. Webcam capture for the live-inference story.

## 7. Conclusion (~0.25 pg)
- 1-paragraph recap of the headline number.
- Future work: RAF-DB run once EULA clears; POSTER++ as 3rd model; cross-dataset transfer.

## 8. References
- DAN: Wen et al. 2021 (`report.md:252`).
- ConvNeXt: Liu et al. 2022.
- EmoNeXt: `report.md:488`.
- FER-2013: Goodfellow et al. 2013 (`report.md:1906`).
- RAF-DB: Li et al. 2017 (`report.md:2186`).

---

## Figures checklist (commit to `figures/`)
- [ ] `class_distribution.png` — §2
- [ ] `cm_dan.png`, `cm_convnext.png`, `cm_dan_vs_convnext.png` — §4.3
- [ ] `perclass_delta.png` — §4.4
- [ ] `agreement_pie.png` — §4.5
- [ ] `failure_gallery_dan.png`, `failure_gallery_convnext.png` — §4.6
- [ ] `gradio_compare_screenshot.png`, `gradio_webcam_screenshot.png` — §6

## Numbers to fill in (search "__" to find them)
- WAR/UAR for DAN, ConvNeXt, both with TTA, ensemble (with and without TTA).
- Train wall-time per model on T4.
- 4 ablation rows (full + 3 removed).
- Agreement counts and oracle upper bound number.
