# Self-Healing Neural Network

### Robust Image Classification Under Severe Corruption

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-16A34A?style=for-the-badge)](LICENSE)

<p align="center">
  <strong>A two-stage deep learning pipeline for CIFAR-100 that restores corrupted images with a Convolutional VAE and classifies the healed output with a ResNet-18 adapted for 32×32 inputs. The repo includes training, evaluation, notebooks, and a React + Express demo UI.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#models-and-training">Models & Training</a> ·
  <a href="#dataset-and-noise">Dataset & Noise</a> ·
  <a href="#web-demo">Web Demo</a> ·
  <a href="#project-structure">Structure</a>
</p>

---

## Overview

Traditional classifiers struggle when inputs are corrupted by blur, Gaussian noise, or salt-and-pepper artifacts. This project addresses that problem with a practical two-stage pipeline:

1. **Stage A - ConvVAE Healer** reconstructs a cleaner image from the corrupted input.
2. **Stage B - ResNet-18 Expert** classifies the healed image after CIFAR-100 normalization.

The pipeline is implemented in [src/pipeline.py](src/pipeline.py) and uses the same CIFAR-100 statistics everywhere in the codebase. The classifier input is normalized after the VAE output is produced, which keeps inference behavior consistent with the training and evaluation code.

### What the repo includes

| Area | What is here |
|---|---|
| Core models | ConvVAE + ResNet-18 classifier wrapper |
| Training | Separate VAE and classifier training scripts |
| Evaluation | Accuracy, top-5, macro F1, ECE, PSNR, SSIM support |
| Notebooks | 5 notebooks covering data, training, integration, and results |
| Web app | Express backend + React/Vite frontend demo |
| Artifacts | Saved checkpoints in `models/` and plots/results in `outputs/` |

---

## Architecture

### End-to-End Flow

```text
┌────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Corrupted CIFAR   │ ──▶ │  ConvVAE Healer      │ ──▶ │  ResNet-18 Expert    │
│  image (32×32×3)   │     │  reconstructs image  │     │  predicts class      │
└────────────────────┘     └──────────────────────┘     └──────────────────────┘
                                      │
                                      ▼
                          Cleaned image in [0,1]
                                      │
                                      ▼
                           CIFAR-100 normalization
                                      │
                                      ▼
                                Class logits
```

### Model Roles

| Stage | Module | Purpose |
|---|---|---|
| A | ConvVAE | Restore a noisy image toward a clean CIFAR-100-like image |
| B | ResNet-18 | Classify the healed image into one of 100 classes |

### ConvVAE at a Glance

The VAE is defined in [src/conv_vae.py](src/conv_vae.py). It uses three downsampling convolution blocks in the encoder, a latent space of 256 dimensions, and a decoder that upsamples back to 32×32 with a sigmoid output layer.

| Component | Current implementation |
|---|---|
| Input | 32×32 RGB image |
| Encoder blocks | 3 convolution blocks |
| Channel progression | 3 → 128 → 256 → 512 |
| Latent dimension | 256 |
| Decoder output | 32×32 RGB image in `[0,1]` |
| Loss | L1 + MSE reconstruction with KL regularization |

### Classifier at a Glance

The classifier is defined in [src/classifier.py](src/classifier.py) and starts from a torchvision ResNet-18 backbone, then swaps in a CIFAR-friendly stem and custom head.

| Component | Current implementation |
|---|---|
| Backbone | ResNet-18 |
| Stem | 3×3 conv, stride 1, no max-pool |
| Output classes | 100 |
| Head | 256-unit hidden layer + dropout + final classifier layer |
| Pretraining | Enabled by default in config |

---

## Models and Training

All runtime settings come from [configs/config.yaml](configs/config.yaml).

### Key Defaults

| Area | Default value |
|---|---|
| Dataset | CIFAR-100 |
| Dataset root | `./data/raw/` |
| Image size | 32 × 32 |
| Class count | 100 |
| Train split | 0.9 |
| Normalization mean | `[0.5071, 0.4865, 0.4409]` |
| Normalization std | `[0.2673, 0.2564, 0.2761]` |
| Train noise types | `gaussian`, `salt_pepper` |
| Train Gaussian std | `0.03` |
| Train salt-pepper prob | `0.005` |
| Eval Gaussian stress | `[0.05, 0.1, 0.15]` |
| VAE latent dim | `256` |
| VAE beta | `0.01` |
| VAE warmup | `20` epochs |
| VAE learning rate | `0.0002` |
| Classifier learning rate | `0.0001` |
| Classifier head LR | `0.001` |
| Clean/healed mix probability | `0.5` |
| Classifier phase 1 | `10` epochs |
| Classifier phase 2 | `50` epochs |

### VAE Training

The VAE training entry point is [src/train_vae.py](src/train_vae.py). It loads CIFAR-100 through the project dataloader, denormalizes inputs to `[0,1]`, and trains the healer with AdamW plus cosine scheduling.

What the script does:

- loads config from `configs/config.yaml`
- resolves the device from the training config
- builds train/validation/test loaders from `src/dataset.py`
- trains the VAE with early stopping and beta warmup
- saves `models/conv_vae_best.pth` and `models/conv_vae_last.pth`

### Classifier Training

The classifier training entry point is [src/train_classifier.py](src/train_classifier.py). It loads the saved VAE, freezes it, and trains the ResNet-18 in two phases.

| Phase | Behavior |
|---|---|
| Phase 1 | Freeze backbone, train the head |
| Phase 2 | Fine-tune the full network |

Training behavior worth knowing:

- the classifier mixes clean normalized images and VAE-healed images with `clean_input_mix_prob`
- validation reports clean and healed branches separately
- the model is saved to `models/resnet_classifier.pth`

### Evaluation

The evaluation entry point is [src/evaluate.py](src/evaluate.py). It supports healing, classification, metric aggregation, and visual summaries.

The evaluation flow reports:

- accuracy
- top-5 accuracy
- macro F1
- expected calibration error
- PSNR
- SSIM

The notebook [05_Evaluation_and_Results.ipynb](notebooks/05_Evaluation_and_Results.ipynb) is the best place to inspect the result workflow and plots.

---

## Dataset and Noise

### Dataset

The project uses CIFAR-100 through torchvision, with the dataset root resolved from the repository root so notebook working directories do not create duplicate data paths.

| Property | Value |
|---|---|
| Dataset | CIFAR-100 |
| Source | `torchvision.datasets.CIFAR100` |
| Image size | 32 × 32 |
| Channels | RGB |
| Classes | 100 |
| Train images | 50,000 |
| Test images | 10,000 |

### Class Names

The class list is defined in [src/dataset.py](src/dataset.py) and includes examples such as `apple`, `aquarium_fish`, `baby`, `bear`, `beaver`, `bottle`, `couch`, `dolphin`, `kangaroo`, and `turtle`.

### Noise Injection

The dataset module provides simple synthetic corruption helpers used by the notebooks and training/evaluation code.

| Noise type | Current config | Notes |
|---|---|---|
| Gaussian | `0.03` train/eval default | Additive noise clipped to `[0,1]` |
| Salt & Pepper | `0.005` train/eval default | Random impulse corruption |
| Gaussian stress sweep | `0.05, 0.1, 0.15` | Evaluation-only sweep |

---

## Web Demo

The web app is split into a Node/Express backend and a React frontend.

### Backend

[web/backend/server.js](web/backend/server.js) exposes a small API surface used by the frontend demo.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Basic service check |
| POST | `/api/classify` | Demo classification response |
| GET | `/api/metrics` | Demo charts and summary metrics |
| GET | `/api/samples` | Sample image paths for the UI |

Important note: the current backend returns demo/mock data for the UI. It is not wired to live Python inference yet.

### Frontend

The frontend lives in [web/frontend/src/App.jsx](web/frontend/src/App.jsx) and uses the component set below.

| Component | Purpose |
|---|---|
| Navbar | Top navigation and site framing |
| Hero | Landing section and project headline |
| Architecture | Pipeline illustration |
| LiveDemo | Upload / demo inference area |
| Metrics | Metrics visualization |
| HowItWorks | Step-by-step explanation |
| TechStack | Stack summary |
| Footer | Closing links |

Frontend tech stack:

- React 18
- Vite
- Framer Motion
- Recharts
- Tailwind CSS

### Run the Web App

Backend:

```bash
cd web/backend
npm install
npm run start
```

Frontend:

```bash
cd web/frontend
npm install
npm run dev
```

---

## Notebooks

The notebook suite is the easiest way to understand the project end-to-end.

| Notebook | Purpose |
|---|---|
| [01_Data_and_Noise.ipynb](notebooks/01_Data_and_Noise.ipynb) | Dataset exploration and corruption demos |
| [02_ConvVAE_Training.ipynb](notebooks/02_ConvVAE_Training.ipynb) | VAE architecture, loss behavior, and training |
| [03_ResNet_Classifier.ipynb](notebooks/03_ResNet_Classifier.ipynb) | Classifier fine-tuning and evaluation |
| [04_Pipeline_Integration.ipynb](notebooks/04_Pipeline_Integration.ipynb) | End-to-end healing + prediction demo |
| [05_Evaluation_and_Results.ipynb](notebooks/05_Evaluation_and_Results.ipynb) | Metrics and results visualization |

Notebook note:

- the notebooks use project-relative paths, so they work best when opened from the repo root
- the pipeline notebook loads the saved checkpoints from `models/`
- the evaluation notebook is the best place to inspect batch metrics and plots

---

## Project Structure

```text
SelfHealingNN/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── models/
│   ├── conv_vae_best.pth
│   ├── conv_vae_epoch_*.pth
│   ├── conv_vae_last.pth
│   └── resnet_classifier.pth
├── notebooks/
│   ├── 01_Data_and_Noise.ipynb
│   ├── 02_ConvVAE_Training.ipynb
│   ├── 03_ResNet_Classifier.ipynb
│   ├── 04_Pipeline_Integration.ipynb
│   └── 05_Evaluation_and_Results.ipynb
├── outputs/
│   ├── plots/
│   └── results/
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── conv_vae.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── train_classifier.py
│   └── train_vae.py
├── web/
│   ├── backend/
│   │   ├── routes/
│   │   │   ├── classify.js
│   │   │   └── metrics.js
│   │   └── server.js
│   └── frontend/
│       └── src/
│           ├── App.jsx
│           ├── components/
│           └── index.css
├── requirements-dev.txt
├── requirements-train.txt
├── setup_training.sh
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ for the web demo
- 8 GB+ RAM recommended
- A GPU is helpful for training, but the project can run on CPU for smaller tests

### Create a Python environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

For GPU training, install the training requirements instead:

```bash
pip install -r requirements-train.txt
```

---

## Running the Project

### Train the VAE

```bash
python -m src.train_vae
```

### Train the classifier

```bash
python -m src.train_classifier
```

### Run evaluation

```bash
python -m src.evaluate
```

### Start the web demo

Backend:

```bash
cd web/backend
npm install
npm run start
```

Frontend:

```bash
cd web/frontend
npm install
npm run dev
```

---

## Results and Artifacts

Training and evaluation produce artifacts under `models/` and `outputs/`.

| Artifact | Description |
|---|---|
| `models/conv_vae_best.pth` | Best saved VAE weights |
| `models/conv_vae_last.pth` | Final VAE checkpoint |
| `models/resnet_classifier.pth` | Final classifier weights |
| `outputs/plots/` | Figures and training visualizations |
| `outputs/results/` | Evaluation outputs and summaries |

If you retrain the project, treat the generated artifacts as the source of truth for any reported numbers.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

- CIFAR-100 via torchvision
- PyTorch for the training and inference stack
- ResNet and VAE research for the architectural baseline

<div align="center">

**Built with PyTorch + React**

[Report Bug](https://github.com/RushikeshMasalkar/SelfHealingNN/issues) | [Request Feature](https://github.com/RushikeshMasalkar/SelfHealingNN/issues)

</div>
