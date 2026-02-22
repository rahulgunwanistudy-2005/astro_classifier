# AstroClassifier

**Deep learning pipeline for galaxy morphology classification — 10 classes, trained from scratch on Galaxy10 DECals.**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/YOUR_USERNAME/astro-classifier)

---

## Live Demo

**[→ Try it on Hugging Face Spaces]()**

Upload any galaxy image and get instant morphology classification with per-class confidence scores.

---

## The Problem

Galaxy morphology classification is a real astrophysics problem — the [Galaxy Zoo project](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) crowdsourced millions of human classifications because it's hard enough that automation is non-trivial.

Two specific challenges make this interesting from an ML perspective:

**1. Class imbalance** — some morphologies are far rarer than others (5.6:1 ratio in this dataset). A naive model that ignores this gets decent accuracy but terrible recall on rare classes.

**2. Visually similar classes** — `unbarred_tight_spiral` and `unbarred_loose_spiral` are so similar that even human annotators disagree. This puts a real ceiling on model performance that hyperparameter tuning alone cannot overcome.

---

## Architecture

Custom CNN trained from scratch — no pretrained backbone.

```
Input (B, 3, 128, 128)
→ ConvBlock(3→32)      # Edge detection
→ ConvBlock(32→64)     # Texture features
→ ConvBlock(64→128)    # Morphological patterns
→ ConvBlock(128→256)   # High-level abstractions
→ GlobalAvgPool        # Translation invariance
→ Classifier(256→128→10)
```

Each ConvBlock: `Conv2d → BatchNorm → ReLU → MaxPool → Dropout`

**0.42M parameters** — deliberately lightweight to demonstrate that architecture and training strategy matter more than scale for this dataset size.

---

## Key Technical Decisions

### Focal Loss (γ=3.0)
Standard cross-entropy ignores class difficulty. Focal Loss down-weights easy examples so the model is forced to learn from hard/rare ones:

```
FL(pₜ) = -α(1 - pₜ)^γ · log(pₜ)
```

### WeightedRandomSampler
Ensures every batch has ~equal class representation regardless of dataset frequency. Without this, the DataLoader rarely shows rare classes even with Focal Loss.

### GlobalAvgPool instead of Flatten
Celestial objects have no canonical orientation. GlobalAvgPool provides translation invariance and reduces parameter count.

### Macro F1 as primary metric
Accuracy is misleading on imbalanced datasets. Macro F1 treats all classes equally.

---

## Dataset

**[Galaxy10 DECals](https://zenodo.org/records/10845026)** — 17,736 galaxy images at 256×256 RGB from the DECam Legacy Survey.

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| disturbed | 756 | 162 | 163 | 1,081 |
| merging | 1,297 | 277 | 279 | 1,853 |
| round_smooth | 1,851 | 396 | 398 | 2,645 |
| in_between | 1,418 | 304 | 305 | 2,027 |
| cigar_shaped | 233 | 50 | 51 | **334** |
| barred_spiral | 1,430 | 306 | 307 | 2,043 |
| unbarred_tight_spiral | 1,280 | 274 | 275 | 1,829 |
| unbarred_loose_spiral | 1,839 | 394 | 395 | 2,628 |
| edge_on_no_bulge | 996 | 213 | 214 | 1,423 |
| edge_on_with_bulge | 1,311 | 280 | 282 | 1,873 |
| **Total** | **12,411** | **2,656** | **2,669** | **17,736** |

Imbalance ratio: **5.6:1** (round_smooth vs cigar_shaped)

---

## Experiment Results

Two training runs, one hypothesis: does higher γ help with the hard classes?

### Run Comparison

| Metric | Run 1 (γ=2.0, 50 ep) | Run 2 (γ=3.0, 75 ep) | Δ |
|--------|----------------------|----------------------|---|
| **Macro F1** | 0.5493 | **0.6069** | +0.058 |
| **Macro AUC** | 0.9211 | **0.9353** | +0.014 |
| **Accuracy** | 58.7% | **64.3%** | +5.6% |

### Per-Class F1 (Run 2 — Best Model)

| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|-----|
| disturbed | 0.286 | 0.399 | 0.333 | 0.796 |
| merging | 0.660 | 0.591 | 0.624 | 0.941 |
| round_smooth | 0.795 | 0.887 | **0.838** | 0.981 |
| in_between | 0.835 | 0.761 | 0.796 | 0.979 |
| cigar_shaped | 0.231 | **0.902** | 0.368 | 0.979 |
| barred_spiral | 0.567 | 0.704 | 0.628 | 0.921 |
| unbarred_tight_spiral | 0.599 | 0.756 | 0.669 | 0.945 |
| unbarred_loose_spiral | 0.640 | 0.139 | 0.229 | 0.854 |
| edge_on_no_bulge | 0.809 | 0.790 | 0.799 | 0.982 |
| edge_on_with_bulge | 0.839 | 0.738 | 0.785 | 0.976 |

### What the results tell us

`cigar_shaped` recall of **0.902** with only 334 training samples is Focal Loss working exactly as intended.

`unbarred_loose_spiral` F1 improved from 0.136 → 0.229 with higher γ, but remains the weak class. This is a **labelling ambiguity problem** not a modelling problem — Galaxy Zoo inter-rater agreement on loose vs tight spirals is historically low. The model has learned a real visual boundary; the ceiling is set by data quality, not architecture.

---

## Project Structure

```
astro-classifier/
├── configs/
│   ├── base_config.yaml          # All hyperparameters
│   └── experiment_focal.yaml     # Run 2 overrides (γ=3.0, 75 epochs)
├── src/
│   ├── data/
│   │   ├── dataset.py            # AstroDataset, augmentations
│   │   └── dataloader.py         # WeightedRandomSampler factory
│   ├── models/
│   │   ├── astro_cnn.py          # Custom CNN architecture
│   │   └── model_factory.py      # Model registry
│   ├── training/
│   │   ├── losses.py             # FocalLoss implementation
│   │   ├── scheduler.py          # Optimizer + LR scheduler
│   │   └── trainer.py            # Training loop, W&B
│   ├── evaluation/
│   │   ├── metrics.py            # Per-class F1, AUC
│   │   ├── evaluator.py          # Full eval pipeline
│   │   └── visualizer.py         # ROC, PR curves, confusion matrix
│   └── utils/
│       ├── config.py             # YAML loader with inheritance
│       ├── logger.py             # Structured logging
│       └── checkpointing.py      # Top-k checkpoint management
├── scripts/
│   ├── prepare_data.py           # Galaxy10 HDF5 → folder structure
│   ├── train.py                  # Training entrypoint
│   ├── evaluate.py               # Standalone evaluation
│   └── infer.py                  # Single image inference
├── demo/
│   └── app.py                    # Gradio demo (HF Spaces)
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory data analysis
└── tests/
    ├── test_dataset.py
    └── test_losses.py
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/astro-classifier.git
cd astro-classifier
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset (~1.4GB)
mkdir -p data/raw
curl -L -o data/raw/Galaxy10_DECals.h5 \
  https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5

# 3. Prepare data
python scripts/prepare_data.py

# 4. Train Run 1
python scripts/train.py --config configs/base_config.yaml

# 5. Train Run 2
python scripts/train.py --config configs/experiment_focal.yaml

# 6. Evaluate
python scripts/evaluate.py \
  --config configs/experiment_focal.yaml \
  --checkpoint outputs/checkpoints_run2/best_model.pth \
  --output-dir outputs/eval_run2

# 7. Run demo locally
pip install -r requirements-demo.txt
python demo/app.py
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Framework | PyTorch 2.0 |
| Loss | Focal Loss (custom) |
| Sampling | WeightedRandomSampler |
| Experiment tracking | Weights & Biases |
| Config system | YAML with inheritance |
| Demo | Gradio → HuggingFace Spaces |
| Dataset | Galaxy10 DECals (Zenodo) |