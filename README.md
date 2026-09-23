<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pytorch-2.2+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/react-19-61dafb?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/fastapi-0.111+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/meta--classifier_accuracy-98.6%25-brightgreen" />
</p>

# RetinAI — Multi-Disease Retinal Classification using EfficientNet-B3 with Anomaly Detection and Meta-Classifier Fusion

A complete end-to-end system for **simultaneous detection of three retinal diseases** from a single fundus photograph: **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Pathologic Myopia (PM)**. The system combines a **diffusion-based anomaly detector** with **disease-specific EfficientNet-B3 CNNs** and a **meta-classifier MLP** that fuses everything into a unified diagnosis — achieving **98.6% routing accuracy** across all three diseases.

> Built by **Ayush Chaudhary** and **Aditya Singh** — Department of Computer Science and Engineering, Amity Centre for Artificial Intelligence (ACAI), Amity University, Noida.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture Overview](#architecture-overview)
  - [Stage 1 — Diffusion Anomaly Detector](#stage-1--diffusion-anomaly-detector)
  - [Stage 2 — Disease-Specific CNNs (4-Channel)](#stage-2--disease-specific-cnns-4-channel)
  - [Stage 3 — Meta-Classifier Fusion](#stage-3--meta-classifier-fusion)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training Pipeline](#training-pipeline)
  - [End-to-End Orchestrator](#end-to-end-orchestrator)
  - [Training Stages Breakdown](#training-stages-breakdown)
  - [Reproducibility](#reproducibility)
- [Inference / Backend API](#inference--backend-api)
- [Frontend (OptiScan AI)](#frontend-optiscan-ai)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
  - [API Usage](#api-usage)
- [Model Weights](#model-weights)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Motivation

- **2.2 billion people** worldwide suffer from vision impairment (WHO).
- **Diabetic Retinopathy**, **Glaucoma**, and **Pathologic Myopia** are major causes of preventable blindness.
- Manual retinal screening is slow, specialist-dependent, and inaccessible in rural/low-resource regions.
- Most existing AI solutions handle **single diseases only**, struggle with **rare or unseen anomalies**, and lack **model interpretability**.

**Our objective**: Build a modular, multi-disease screening system from a single retinal fundus image — with explainable attention maps that clinicians can trust.

---

## Architecture Overview

```
INPUT: Retinal Fundus Image (512×512×3)
                │
                ▼
    ┌───────────────────────┐
    │   DIFFUSION MODEL     │
    │   (UNet2D - DDPM)     │
    │   Trained on Healthy  │
    │   Retinas Only        │
    └───────────┬───────────┘
                │
      ┌─────────┴─────────┐
      ▼                   ▼
 Attention Map      Anomaly Score
  (256×256×1)         (scalar)
      │                   │
      ▼                   │
  CONCATENATE ────────────┘
  [R,G,B] + [AnomalyMap]
  → 4-Channel Tensor
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  CNN-1   │  │  CNN-2   │  │  CNN-3   │
 │   DR     │  │ Glaucoma │  │   PM     │
 │EffNet-B3 │  │EffNet-B3 │  │EffNet-B3 │
 │ 5 class  │  │ 2 class  │  │ 2 class  │
 └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │
      │   1536-dim   │   1536-dim   │   1536-dim
      │   backbone   │   backbone   │   backbone
      │   features   │   features   │   features
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   META-CLASSIFIER   │
          │   MLP (4609 → 3)    │
          │                     │
          │  [1536 + 1536 +     │
          │   1536 + 1 anomaly] │
          │       = 4609 dim    │
          └──────────┬──────────┘
                     │
                     ▼
          Disease Classification
        + Confidence Scores
        + Attention Maps
```

### Stage 1 — Diffusion Anomaly Detector

A **Denoising Diffusion Probabilistic Model (DDPM)** trained exclusively on **healthy retinal images**. At inference, pathological images produce higher reconstruction error, which becomes the anomaly signal.

| Parameter | Value |
|-----------|-------|
| Architecture | UNet2DModel (HuggingFace Diffusers) |
| Input Resolution | 256 × 256 |
| Channels | `(128, 256, 512, 512)` with attention in deeper blocks |
| Noise Timesteps | 1000 (training), 500 (inference) |
| N Reconstruction Samples | 3 (inference) / 5 (training) |
| Training Epochs | 60 |
| Optimizer | AdamW, LR = 1e-4 |
| Mixed Precision | FP16 via `torch.amp` |

**How it works**: The diffusion reconstruction cannot faithfully reconstruct pathological structures (microaneurysms, optic disc cupping, posterior staphyloma) it has never seen → the pixel-wise residual becomes an **anomaly/attention map**.

### Stage 2 — Disease-Specific CNNs (4-Channel)

Three independent **EfficientNet-B3** classifiers, each receiving a **4-channel input** `[R, G, B, AnomalyMap]`:

| Model | Task | Classes | Output |
|-------|------|---------|--------|
| **CNN-1** | Diabetic Retinopathy severity | 5 (No DR, Mild, Moderate, Severe, Proliferative) | 5-class softmax |
| **CNN-2** | Glaucoma detection | 2 (Non-Glaucoma, Glaucoma) | Binary softmax |
| **CNN-3** | Pathologic Myopia detection | 2 (Non-PM, PM) | Binary softmax |

**Key implementation details**:
- **4-channel stem**: The first convolutional layer is modified to accept 4 input channels. Weights for RGB channels are initialized from ImageNet pretrained values; the anomaly channel is initialized with 1% of the first channel weights.
- **Training**: ImageNet pretrained backbone, CosineAnnealingLR scheduler, weighted cross-entropy loss for class imbalance, early stopping.
- **Augmentation** (for 4-ch `.npy` tensors): Horizontal flip, vertical flip, ±15° rotation — applied consistently across all 4 channels.

### Stage 3 — Meta-Classifier Fusion

A lightweight **MLP** that fuses the three CNN backbones into a single multi-disease prediction:

```
Input:  [CNN-1 features (1536) | CNN-2 features (1536) | CNN-3 features (1536) | anomaly_score (1)]
        = 4609-dimensional vector

Architecture:
  Linear(4609 → 512) → BatchNorm → ReLU → Dropout(0.3)
  Linear(512 → 256)  → BatchNorm → ReLU → Dropout(0.3)
  Linear(256 → 3)    → 3-class output [DR, Glaucoma, PM]
```

The meta-classifier learns **cross-disease correlations** and produces a unified routing decision with calibrated confidence scores.

---

## Key Results

### Performance Comparison of Disease-Specific EfficientNet-B3 Models

| Model | Task | Accuracy | Macro F1 | Weighted F1 | AUC-ROC |
|-------|------|----------|----------|-------------|---------|
| **CNN-1** | Diabetic Retinopathy (5-class) | 0.622 | 0.538 | 0.613 | 0.809 |
| **CNN-2** | Glaucoma Detection | **0.951** | 0.928 | 0.951 | 0.940 |
| **CNN-3** | Pathologic Myopia Detection | **0.965** | 0.965 | 0.965 | **0.995** |
| **Meta-Classifier** | Multi-Disease Routing (3-class) | **0.986** | 0.970 | 0.986 | **0.998** |

### Meta-Classifier Confusion Matrix

|  | Predicted DR | Predicted Glaucoma | Predicted PM |
|--|-------------|-------------------|-------------|
| **Actual DR** | **2025** | 0 | 0 |
| **Actual Glaucoma** | 7 | **355** | 9 |
| **Actual PM** | 8 | 16 | **376** |

### Highlights

- **98.6%** meta-classifier routing accuracy across three retinal diseases
- **100% recall** for DR detection — no diabetic retinopathy cases missed
- **99.5% AUC-ROC** for Pathologic Myopia — near-perfect discrimination
- **99.8% AUC-ROC** for the meta-classifier ensemble
- DR grading (5-class) remains the most challenging task due to subtle inter-grade differences (62.2% accuracy, 80.9% AUC)
- Strong performance for Glaucoma (~95% F1) and PM (~95-96% F1)

---

## Project Structure

```
RetinAI/
├── backend/                          # FastAPI inference server
│   ├── main.py                       # API endpoints (/health, /predict)
│   ├── requirements.txt              # Python dependencies
│   ├── models/
│   │   ├── architectures.py          # EfficientNet-B3 (4ch) + MetaMLP definitions
│   │   ├── inference_real.py         # Full 4-stage inference pipeline
│   │   └── inference.py              # Demo/placeholder inference
│   └── weights/                      # Model checkpoints (gitignored)
│       ├── diffusion_best.pt         # ~1.2 GB  — Diffusion UNet
│       ├── cnn1_dr_best.pth          # ~123 MB  — DR classifier
│       ├── cnn2_glaucoma_best.pth    # ~123 MB  — Glaucoma classifier
│       ├── cnn3_pm_best.pth          # ~123 MB  — PM classifier
│       └── meta_classifier_best.pth  # ~10 MB   — Meta-classifier MLP
│
├── Frontend/                         # React + TypeScript web app
│   ├── src/
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx       # Marketing page with scroll animations
│   │   │   └── MainApp.tsx           # Application shell (upload → results)
│   │   ├── sections/                 # Landing page sections (Hero, Pipeline, etc.)
│   │   └── components/               # Shared UI components (Navigation, shadcn/ui)
│   ├── package.json
│   └── vite.config.ts
│
├── training/                         # Full training pipeline
│   ├── run_on_5090.py                # End-to-end orchestrator (6 stages)
│   ├── train_cnn1_dr.py              # CNN-1: DR training with LR sweep
│   ├── train_cnn2_glaucoma.py        # CNN-2: Glaucoma training
│   ├── train_cnn3_pm.py              # CNN-3: PM training
│   ├── train_meta_classifier.py      # Meta-classifier training
│   ├── train_diffusion.py            # Diffusion model training (256px)
│   ├── evaluate_models.py            # Single model evaluation
│   ├── evaluate_all.py               # Full pipeline evaluation
│   ├── generate_4ch_eyepacs.py       # 4-channel data generation (EyePACS)
│   ├── generate_4ch_palm.py          # 4-channel data generation (PALM)
│   ├── generate_anomaly_maps.py      # Anomaly map batch generation
│   ├── prepare_glaucoma_combined.py  # Combine REFUGE + G1020 + ORIGA datasets
│   ├── test_run.py                   # Quick pipeline sanity test
│   ├── src/                          # Training library
│   │   ├── datasets/                 #   RetinalDataset, EyePACS, REFUGE, PALM
│   │   ├── models/                   #   EfficientNet-B3 architecture (training copy)
│   │   ├── training/                 #   Trainer class (AMP, checkpointing)
│   │   ├── evaluation/               #   Metrics (accuracy, F1, AUC, confusion matrices)
│   │   └── utils/                    #   Transforms, reproducibility, checkpoint utils
│   └── diffusion/                    # 512px diffusion variant
│       ├── train_diffusion.py        # High-res diffusion training
│       ├── evaluate_diffusion.py     # Diffusion evaluation
│       ├── preprocess.py             # Healthy image preprocessing
│       ├── split.py                  # Train/val splitting
│       └── generate_anomaly_maps.py  # Batch anomaly map generation
│
├── evaluation_results/               # Saved evaluation outputs
│   ├── all_results.json              # Full metrics for all models
│   └── evaluation_results.log        # Evaluation logs
│
├── tests/                            # Test scaffolding
│   ├── unit/
│   ├── integration/
│   └── property/
│
├── docs/
│   └── retinai_dashboard.html        # Visualization dashboard
│
├── RUNNING.md                        # Quick-start guide
├── .gitignore
└── README.md                         # ← You are here
```

---

## Datasets

### 1. EyePACS — Diabetic Retinopathy

| Property | Value |
|----------|-------|
| **Source** | Kaggle Diabetic Retinopathy Detection Challenge (2015) |
| **Images** | 88,702 color fundus photographs |
| **Classes** | 5: No DR (Grade 0), Mild, Moderate, Severe, Proliferative DR (Grade 4) |
| **Resolution** | 433×289 to 5184×3456 pixels |

### 2. Glaucoma Combined (REFUGE + G1020 + ORIGA)

| Property | Value |
|----------|-------|
| **Source** | REFUGE Challenge (ISBI 2018), G1020, ORIGA datasets |
| **Images** | ~1,200 (expanded with additional data) |
| **Classes** | 2: Non-Glaucoma, Glaucoma |
| **Resolution** | 2124×2056 (Zeiss) and 1634×1634 (Canon) |
| **Preparation** | `prepare_glaucoma_combined.py` merges and standardizes all three sources |

### 3. PALM — Pathologic Myopia

| Property | Value |
|----------|-------|
| **Source** | ISBI 2019 PALM Challenge (Multi-center China) |
| **Images** | 1,200 (800 Non-PM + 400 PM) |
| **Classes** | 2: Non-PM, Pathologic Myopia |
| **Resolution** | Variable, 45° field-of-view |

### 4-Channel Data Generation

The diffusion model's anomaly maps are precomputed and concatenated with the original RGB images to create **4-channel `.npy` tensors** `[R, G, B, AnomalyMap]`. This is done per-dataset:

```bash
python training/generate_4ch_eyepacs.py --checkpoint <diffusion_best.pt> --input_dir <eyepacs> --output_dir <eyepacs_4ch>
python training/generate_4ch_palm.py    --checkpoint <diffusion_best.pt> --input_dir <palm> --output_dir <palm_4ch>
```

---

## Training Pipeline

### End-to-End Orchestrator

`training/run_on_5090.py` runs the **complete 6-stage pipeline** in sequence:

```
Stage 0 → GPU check (CUDA, VRAM, driver version)
Stage 1 → Install dependencies (torch, diffusers, timm, etc.)
Stage 2A → Generate 4-channel data for EyePACS (DR)
Stage 2B → Generate 4-channel data for PALM (PM)
Stage 3 → Train CNN-1 — Diabetic Retinopathy (100 epochs, LR=4e-4)
Stage 4 → Train CNN-2 — Glaucoma (100 epochs, LR=3e-4)
Stage 5 → Train CNN-3 — Pathologic Myopia (100 epochs, LR=3e-4)
Stage 6 → Train Meta-Classifier (stacks all 3 CNN checkpoints)
```

**Auto-resume**: Each checkpoint is inspected; if `val_acc_history` has ≥50 epochs, that stage is automatically skipped — enabling crash recovery.

### Training Stages Breakdown

#### Diffusion Model (trained separately)

```bash
cd training
python train_diffusion.py
```

- Trains UNet2DModel (DDPM, 1000 timesteps) on healthy-only fundus images
- 60 epochs, batch size 4, AdamW LR 1e-4, FP16 mixed precision
- Early stopping with patience of 10 epochs
- Used to generate anomaly maps for the CNN training data

#### CNN Training (each disease)

```bash
python training/train_cnn1_dr.py --labels_file <csv> --output_dir models/checkpoints/dr
python training/train_cnn2_glaucoma.py --labels_file <csv> --output_dir models/checkpoints/glaucoma
python training/train_cnn3_pm.py --labels_file <csv> --output_dir models/checkpoints/pm
```

Common settings across all three CNNs:
- **Backbone**: EfficientNet-B3 (ImageNet pretrained, 4-channel input)
- **Epochs**: 100 with CosineAnnealing LR scheduler
- **Loss**: Weighted cross-entropy (handles class imbalance)
- **Augmentation**: Horizontal/vertical flip, ±15° rotation (applied to 4-ch tensors)
- **Batch size**: 16–32
- **Seed**: 42

#### Meta-Classifier Training

```bash
python training/train_meta_classifier.py \
  --cnn1_checkpoint <dr_best.pth> \
  --cnn2_checkpoint <glaucoma_best.pth> \
  --cnn3_checkpoint <pm_best.pth> \
  --labels_dr <eyepacs_csv> \
  --labels_glaucoma <glaucoma_csv> \
  --labels_pm <palm_csv>
```

Extracts frozen 1536-dim backbone features from each CNN + anomaly score → trains a lightweight 3-layer MLP.

### Reproducibility

All training scripts call `set_random_seeds(42)` which seeds Python's `random`, NumPy, PyTorch (CPU + CUDA), and sets `cudnn.deterministic=True`:

```python
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Inference / Backend API

The FastAPI backend (`backend/main.py`) loads all five model weights at startup and exposes a REST API.

### Inference Pipeline (4 stages)

```python
# Stage 1: Diffusion anomaly detection
attention_map, anomaly_score = diffusion_model.reconstruct(image)

# Stage 2: 4-channel CNN inference
four_ch = concat([R, G, B, attention_map])
dr_logits       = cnn1_dr(four_ch)         # (1, 5)
glaucoma_logits = cnn2_glaucoma(four_ch)    # (1, 2)
myopia_logits   = cnn3_pm(four_ch)          # (1, 2)

# Stage 3: Feature extraction + Meta-classifier
dr_feats = extract_backbone(cnn1, four_ch)       # 1536-dim
gl_feats = extract_backbone(cnn2, four_ch)       # 1536-dim
pm_feats = extract_backbone(cnn3, four_ch)       # 1536-dim
meta_input = concat([dr_feats, gl_feats, pm_feats, anomaly_score])  # 4609-dim
meta_output = meta_mlp(meta_input)               # (1, 3)

# Stage 4: Fuse CNN probabilities with meta-classifier confidence
final_predictions = fuse(cnn_probs, meta_probs)
```

### API Endpoints

#### `GET /health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "input_size": 256
}
```

#### `POST /predict`

Upload a retinal fundus image (JPEG/PNG, max 50 MB).

**Response**:
```json
{
  "scan_id": "uuid",
  "inference_ms": 1250,
  "anomaly_score": 0.42,
  "predictions": {
    "diabetic_retinopathy": {
      "probability": 0.85,
      "severity": "Moderate",
      "description": "Moderate diabetic retinopathy detected..."
    },
    "glaucoma": {
      "probability": 0.12,
      "severity": "Low Risk",
      "description": "..."
    },
    "pathologic_myopia": {
      "probability": 0.08,
      "severity": "Low Risk",
      "description": "..."
    }
  },
  "meta": {
    "primary_diagnosis": "diabetic_retinopathy",
    "primary_probability": 0.85,
    "risk_level": "High"
  },
  "attention_map_b64": "<base64 PNG heatmap>"
}
```

---

## Frontend (OptiScan AI)

A modern React + TypeScript web application with:

- **Landing page** (`/`) — Animated scroll experience with GSAP/ScrollTrigger showcasing the pipeline, live mockups of the upload/analysis/results flow, and key statistics
- **Application** (`/app`) — Full clinical dashboard with:
  - **Upload & Analysis** (`/app/health`) — Drag-and-drop fundus image upload, real-time API inference, attention heatmap overlay, per-disease probability bars, risk level assessment, severity descriptions, and scan metadata
  - **Dashboard** — Disease selection cards, quick actions, patient search
  - **Schedule** — Appointment calendar (week view)
  - **Reports** — Historical analysis reports
  - **Settings** — Dark mode toggle, account preferences

**Key technologies**: React 19, Vite, TypeScript, Tailwind CSS, shadcn/ui (Radix primitives), GSAP animations, Recharts, React Router v7.

---

## Getting Started

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **CUDA-capable GPU** (recommended for inference; CPU works but is slower)
- ~**1.6 GB** disk space for model weights

### Installation

```bash
# Clone the repository
git clone https://github.com/YayaBud/RetinAI.git
cd RetinAI

# ── Backend ──
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
cd ..

# ── Frontend ──
cd Frontend
npm install
cd ..
```

### Model Weights

Place the five checkpoint files in `backend/weights/`:

```
backend/weights/
├── diffusion_best.pt         # ~1.2 GB
├── cnn1_dr_best.pth          # ~123 MB
├── cnn2_glaucoma_best.pth    # ~123 MB
├── cnn3_pm_best.pth          # ~123 MB
└── meta_classifier_best.pth  # ~10 MB
```

> **Download the pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1g9D6Ca4do816uyvefzbYWWS6HVuCmGn-?usp=sharing)** and place them in `backend/weights/`. They are not included in the repository due to size.

### Running the Application

**Terminal 1 — Backend** (FastAPI on port 8000):

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend** (Vite dev server on port 5173):

```bash
cd Frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Run prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/fundus_image.jpg"
```

```python
# Python
import httpx

with open("fundus.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/predict",
        files={"file": ("fundus.jpg", f, "image/jpeg")}
    )
    result = response.json()
    print(result["meta"]["primary_diagnosis"])
    print(result["predictions"])
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Deep Learning** | PyTorch 2.2+, HuggingFace Diffusers, timm (EfficientNet-B3) |
| **Backend API** | FastAPI, Uvicorn, Pillow, NumPy |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS, shadcn/ui, GSAP, Recharts |
| **Training Infra** | Mixed-precision (FP16), CosineAnnealingLR, early stopping, auto-resume |
| **Evaluation** | scikit-learn (F1, AUC-ROC, confusion matrices) |
| **Reproducibility** | Fixed seeds (42), deterministic CUDA, structured checkpointing |

---

## License

This project was developed as part of academic research at **Amity Centre for Artificial Intelligence (ACAI), Amity University**. Please contact the author for licensing inquiries.
