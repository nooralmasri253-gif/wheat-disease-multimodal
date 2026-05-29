# Wheat Disease Multimodal Classification

**Course:** Deep Learning — Spring 2026, University of Jordan
**Instructor:** Dr. Tamam AlSarhan
**Competition:** [ICPR 2026 Beyond Visible Spectrum: AI for Agriculture](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026)
**Individual submission**

> 3-class wheat disease classification (`Health` / `Rust` / `Other`) from co-registered UAV imagery using a three-way modality ablation: RGB → RGB+MS → RGB+MS+HS.

---

## Results Summary

| Model | CV Accuracy | Kaggle Public | Kaggle Private |
|-------|-------------|---------------|----------------|
| RGB only (baseline) | 0.598 | 0.505 | 0.624 |
| RGB + MS | 0.640 | 0.632 | 0.649 |
| **RGB + MS + HS (final)** | **0.673** | **0.684** | **0.722** |

---

## Repository Structure

```
wheat-disease-multimodal/
├── README.md                        # This file
├── requirements.txt                 # Pinned dependencies
├── .gitignore
├── Wheat_Disease_Notebook.ipynb     # All code: EDA + training + evaluation + submission
├── Wheat_Disease_Report.docx        # Full written report (13 pages)
└── Wheat_Disease_Slides.pptx        # Presentation slides (11 slides)
```

---

## How to Reproduce

### Option A — Kaggle (recommended, free GPU)

1. Go to the [competition page](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026) and accept the rules
2. Open a new Kaggle Notebook and attach the competition dataset as input
3. Upload `Wheat_Disease_Notebook.ipynb` via **File → Import Notebook**
4. Enable GPU: right sidebar → **Accelerator → GPU T4 x2**
5. Run all cells top to bottom (**~35 minutes** on T4 GPU)
6. Download `submission.csv` from the Output panel and submit to the competition

### Option B — Local machine

```bash
# 1. Clone
git clone https://github.com/<your-username>/wheat-disease-multimodal.git
cd wheat-disease-multimodal

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Download data
# From https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026/data
# Unzip so the structure is:
#   data/Kaggle_Prepared/train/{HS,MS,RGB}/
#   data/Kaggle_Prepared/val/{HS,MS,RGB}/

# 4. Edit DATA_ROOT in Cell 3 of the notebook to:
#   DATA_ROOT = Path('data/Kaggle_Prepared')

# 5. Run the notebook
jupyter notebook Wheat_Disease_Notebook.ipynb
# Then run all cells in order (Kernel → Restart & Run All)
```

---

## Dependencies

All dependencies are pinned in `requirements.txt`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.3.0 | Deep learning framework |
| timm | 1.0.7 | Pretrained ResNet50 backbone |
| torchvision | 0.18.0 | Image transforms |
| rasterio | 1.3.10 | Loading hyperspectral/multispectral TIFF files |
| scikit-learn | 1.5.0 | Stratified CV, classification metrics |
| numpy | 1.26.4 | Numerical operations |
| seaborn | 0.13.2 | Confusion matrix visualization |

---

## Notebook Structure

`Wheat_Disease_Notebook.ipynb` contains the complete pipeline in 15 cells:

| Cell | Description |
|------|-------------|
| 1 | Install dependencies (pinned versions) |
| 2 | Reproducibility — fixed seeds (Python, NumPy, PyTorch, cuDNN) |
| 3 | Configuration — all paths and hyperparameters in one place |
| 4 | Build training dataframe |
| 5 | EDA — one sample per class per modality |
| 6 | EDA — mean hyperspectral signatures per class (key diagnostic figure) |
| 7 | Per-band normalization statistics for HS and MS |
| 8 | Tri-modal Dataset class (RGB + MS + HS) |
| 9 | Model — ResNet50 with HSSpectralEncoder novelty |
| 10 | Training and validation functions |
| 11 | **5-fold stratified cross-validation training (~35 min)** |
| 12 | Out-of-fold confusion matrix + per-class classification report |
| 13 | Generate Kaggle submission (5-fold soft-vote ensemble) |
| 14 | Per-fold performance bar chart |
| 15 | Final summary printout |

---

## Model Architecture

```
RGB  (3ch,  64×64)  →  resize 224×224  →  ImageNet normalize ─┐
MS   (5ch,  64×64)  →  resize 224×224  →  z-score normalize  ─┤→ stack (8ch, 224×224)
                                                                │
HS (125ch, 32×32)  →  HSSpectralEncoder:                       │
                        Conv1D 125→64→16 (along spectral axis)  │
                        Bilinear upsample 32×32 → 224×224      →┘ (16ch, 224×224)

Concatenated: (24ch, 224×224) → ResNet50 (in_chans=24) → 3 classes
```

**HSSpectralEncoder** adds ~10,000 parameters and compresses 125 HS bands into 16
task-relevant channels using 1×1 convolutions along the spectral axis. This is a
learned alternative to PCA — it optimizes for classification, not variance.

---

## Experimental Design

Three models trained under **identical conditions** (same backbone, optimizer, augmentation,
seeds, 5-fold CV) differing only in input modality:

- **Model A — RGB only:** 3-channel input, establishes the visible-spectrum baseline
- **Model B — RGB + MS:** 8-channel input, adds Red Edge (~740nm) and NIR (~833nm)
- **Model C — RGB + MS + HS:** 24-channel input, adds full 125-band spectral cube

This controlled ablation allows causal attribution: any performance difference is due
to the added spectral information, not to architecture or training differences.

---

## Reproducibility Notes

- Seeds fixed: `SEED = 42` for Python `random`, NumPy, PyTorch CPU/GPU, cuDNN deterministic
- Per-fold seeds: `SEED + fold` (different augmentation streams, same overall experiment)
- Expected variance across reruns: ±0.5% accuracy
- Training time: ~35 minutes total on Kaggle T4 GPU

---

## Dataset

Download from the [competition data page](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026/data).

| Split | Health | Rust | Other | Total |
|-------|--------|------|-------|-------|
| Train | 200 | 200 | 200 | **600** |
| Val / Test | — | — | — | 300 (labels hidden) |

The validation set labels are held by Kaggle and used for leaderboard scoring.
Our Kaggle private score (0.722) is the test set result.
