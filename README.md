# Self-Healing Neural Network

Robust image classification for CIFAR-100 using a two-stage pipeline: a ConvVAE restores corrupted images, then a ResNet-18 classifier predicts the class from the healed output.

[Python 3.10+](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org) [![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev) [![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com) [![License](https://img.shields.io/badge/License-MIT-16A34A?style=flat-square)](LICENSE)

## Overview

The repository contains:

- a Python pipeline for noisy image healing and classification in [src/pipeline.py](src/pipeline.py)
- training entry points for the VAE and classifier in [src/train_vae.py](src/train_vae.py) and [src/train_classifier.py](src/train_classifier.py)
- an evaluation module that reports accuracy, top-5 accuracy, macro F1, ECE, PSNR, and SSIM in [src/evaluate.py](src/evaluate.py)
- a small Express backend and React/Vite frontend for the web demo in [web/backend/server.js](web/backend/server.js) and [web/frontend/src/App.jsx](web/frontend/src/App.jsx)

The default dataset is CIFAR-100. Dataset paths are resolved from the repository root, so notebooks and scripts can be run from different working directories without creating duplicate data folders.

## Architecture

Pipeline flow:

`corrupted input -> ConvVAE healer -> normalized classifier input -> ResNet-18 prediction`

Implementation details:

- The VAE reconstructs images in the `[0, 1]` range and is trained on noisy/clean image pairs.
- The classifier uses a pretrained ResNet-18 with a CIFAR-friendly stem: 3x3 stride-1 convolution, no initial max-pool, and a custom MLP head.
- The pipeline normalizes the VAE output before classification, which matches the behavior in [src/pipeline.py](src/pipeline.py).

## Key Defaults

Values below come from [configs/config.yaml](configs/config.yaml).

| Area | Default |
|---|---|
| Dataset | CIFAR-100 |
| Image size | 32 x 32 |
| Dataset root | `./data/raw/` |
| Normalization | CIFAR-100 mean/std |
| Noise types | `gaussian`, `salt_pepper` |
| Gaussian noise | `0.03` for training/eval config |
| Salt & pepper | `0.005` for training/eval config |
| VAE latent dim | `256` |
| VAE beta | `0.01` |
| VAE warmup | `20` epochs |
| VAE learning rate | `0.0002` |
| Classifier backbone | `resnet18` |
| Classifier learning rate | `0.0001` |
| Classifier head LR | `0.001` |
| Clean/healed mix prob | `0.5` |
| Phase 1 epochs | `10` |
| Phase 2 epochs | `50` |

## Repository Layout

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
│   ├── classifier.py
│   ├── conv_vae.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── train_classifier.py
│   └── train_vae.py
└── web/
    ├── backend/
    │   ├── server.js
    │   └── routes/
    │       ├── classify.js
    │       └── metrics.js
    └── frontend/
        └── src/
            ├── App.jsx
            └── components/
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ for the web demo
- 8 GB+ RAM recommended
- Optional GPU for training

### Create an environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

For GPU training, use:

```bash
pip install -r requirements-train.txt
```

## Training

From the repository root:

```bash
python -m src.train_vae
python -m src.train_classifier
```

The training scripts write checkpoints and metrics to `models/` and `outputs/results/`.

Expected artifacts:

- `models/conv_vae_best.pth`
- `models/conv_vae_last.pth`
- `models/resnet_classifier.pth`

## Evaluation

Run the evaluation entry point after both models are available:

```bash
python -m src.evaluate
```

The evaluation module reports:

- accuracy
- top-5 accuracy
- macro F1
- expected calibration error
- PSNR and SSIM for reconstruction quality

The notebook [05_Evaluation_and_Results.ipynb](notebooks/05_Evaluation_and_Results.ipynb) is the easiest place to inspect the full set of plots and summaries.

## Web Demo

The web app is split into a small Express backend and a React frontend.

### Backend

```bash
cd web/backend
npm install
npm run start
```

Available endpoints:

- `GET /api/health`
- `POST /api/classify`
- `GET /api/metrics`
- `GET /api/samples`

The current backend returns demo data for UI testing, not live Python inference.

### Frontend

```bash
cd web/frontend
npm install
npm run dev
```

The frontend uses React, Vite, Framer Motion, Recharts, and Tailwind CSS.

## Notebooks

| Notebook | Purpose |
|---|---|
| [01_Data_and_Noise.ipynb](notebooks/01_Data_and_Noise.ipynb) | Dataset exploration and noise visualization |
| [02_ConvVAE_Training.ipynb](notebooks/02_ConvVAE_Training.ipynb) | VAE training and reconstruction analysis |
| [03_ResNet_Classifier.ipynb](notebooks/03_ResNet_Classifier.ipynb) | Classifier training and fine-tuning |
| [04_Pipeline_Integration.ipynb](notebooks/04_Pipeline_Integration.ipynb) | End-to-end healing plus prediction |
| [05_Evaluation_and_Results.ipynb](notebooks/05_Evaluation_and_Results.ipynb) | Metrics, plots, and comparison tables |

If a notebook kernel starts in a different working directory, keep paths anchored to the repository root or use the path resolution already built into the training code.

## Configuration Notes

- [configs/config.yaml](configs/config.yaml) controls dataset paths, noise settings, model hyperparameters, and evaluation sweeps.
- The pipeline normalizes classifier inputs with CIFAR-100 mean/std before prediction.
- The VAE decoder uses a sigmoid output layer, so reconstructed pixels are in `[0, 1]` before normalization.

## Results Output

Evaluation summaries are written under `outputs/results/` and include per-condition metrics, confusion matrices, and protocol summaries.

If you retrain the models, treat the generated files in `outputs/results/` and `models/` as the source of truth for reported numbers.

## License

MIT License. See [LICENSE](LICENSE).
