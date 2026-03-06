<div align="center">

# Self-Healing Neural Network

### A Two-Stage Deep Learning Pipeline for Catastrophic Noise Recovery & Classification

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

**A neural network that doesn't just classify — it _heals_ corrupted data first.**

Instead of feeding noisy images directly into a classifier, this architecture intercepts corrupted data,<br/>
mathematically filters out the noise using a generative model, and then classifies the reconstructed output.

<br/>

| Clean Input | Heavy Noise Applied | VAE Healed Output | CNN Prediction |
|:-----------:|:-------------------:|:-----------------:|:--------------:|
| Original | Corrupted | Recovered | Classified |

<br/>

[Explore the Notebooks](#notebook-pipeline) · [Run the Dashboard](#quick-start) · [View Benchmarks](#performance--benchmarks)

</div>

<br/>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Performance & Benchmarks](#performance--benchmarks)
- [Interactive Dashboard](#interactive-dashboard)
- [Quick Start](#quick-start)
- [Notebook Pipeline](#notebook-pipeline)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

---

## Overview

Standard image classification models **degrade rapidly** when exposed to noisy, real-world data. A CNN trained to 98%+ accuracy on clean images can plummet below 50% when Gaussian noise is introduced — a phenomenon known as **catastrophic model collapse**.

This project solves that problem with a **Self-Healing AI pipeline**: a two-stage architecture that **mathematically isolates** the denoising task from the classification task.

### How It Works

```
┌──────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌────────────┐
│              │      │   NOISE LAYER   │      │  HEALER (VAE)   │      │EXPERT (CNN)│
│  Clean Input │ ───► │  Gaussian Noise │ ───► │   Denoising &   │ ───► │Classification│
│   28×28 px   │      │  σ = 0.0 – 1.0  │      │  Reconstruction │      │  10 Classes │
│              │      │                 │      │                 │      │            │
└──────────────┘      └─────────────────┘      └─────────────────┘      └────────────┘
                                                       │
                                          Reparameterization Trick
                                            z = μ + σ ⊙ ε
```

> **Core Objective:** Prevent model collapse in catastrophically degraded environments by separating noise reduction from classification into two specialized, jointly optimized networks.

---

## Key Results

<div align="center">

| Scenario | Accuracy | Change |
|:---|:---:|:---:|
| **Baseline** — CNN on clean data | **98.91%** | — |
| **Disaster** — CNN on noisy data | **47.92%** | −51.0% |
| **Self-Healing Pipeline** (fine-tuned) | **93.82%** | **+45.9%** recovery |

</div>

> **The Self-Healing pipeline recovered +45.90 percentage points** of accuracy compared to a standalone CNN under heavy Gaussian noise — restoring near-baseline performance in catastrophic conditions.

---

## System Architecture

The pipeline consists of two independent neural networks trained sequentially and later **fine-tuned end-to-end** to achieve joint optimization.

### Stage A — The "Healer" (Variational Autoencoder)

A **generative probabilistic model** responsible for strict denoising.

| Component | Function |
|:---|:---|
| **Encoder** | Compresses noisy input spatially via sequential `Conv2d` layers |
| **Latent Space** | Maps features to a continuous distribution using the Reparameterization Trick, forcing the network to discard random Gaussian static while retaining core structural features |
| **Decoder** | Upsamples the latent vector via `ConvTranspose2d` back to a clean image space |

**Reparameterization Trick:**

$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$

**Loss Function:** Dual-objective combining **Binary Cross-Entropy** (reconstruction) and **KL-Divergence** (regularization):

$$\mathcal{L}_{VAE} = \text{BCE}(x, \hat{x}) + D_{KL}\big(\mathcal{N}(\mu, \sigma^2) \;\|\; \mathcal{N}(0, 1)\big)$$

### Stage B — The "Expert" (Deep CNN)

A **Convolutional Neural Network** trained strictly on pristine, uncorrupted data. It acts as the baseline classifier, protected from real-world noise by the Healer's preprocessing.

| Layer | Details |
|:---|:---|
| Conv Block 1 | `Conv2d(1→32)` → `ReLU` → `MaxPool2d(2×2)` |
| Conv Block 2 | `Conv2d(32→64)` → `ReLU` → `MaxPool2d(2×2)` |
| Classifier | `Linear(3136→128)` → `ReLU` → `Linear(128→10)` |

### End-to-End Joint Optimization

The true power was unlocked by combining both computational graphs into a **single differentiable pipeline**:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{VAE}} + \lambda \cdot \mathcal{L}_{\text{CNN}}$$

This forces the Healer to preserve the **exact geometric features** the Expert requires for classification — not just produce visually clean images.

---

## Performance & Benchmarks

### Accuracy Under Catastrophic Gaussian Noise

```
 100% ┤
  98% ┤ ████                                          Baseline (Clean): 98.91%
      ┤ ████
  93% ┤ ████                           ████           Self-Healing:     93.82%
      ┤ ████                           ████
      ┤ ████                           ████
      ┤ ████                           ████
      ┤ ████                           ████
      ┤ ████                           ████
      ┤ ████            ████           ████
  48% ┤ ████            ████           ████            CNN + Noise:      47.92%
      ┤ ████            ████           ████
      ┼─────────────────────────────────────
        Baseline        Disaster       Self-Healing
       (Clean Data)   (CNN + Noise)    (Fine-Tuned)
```

| Metric | Value |
|:---|:---:|
| Accuracy recovered from noise collapse | **+45.90%** |
| Final pipeline accuracy (noisy input) | **93.82%** |
| Baseline accuracy (clean input) | **98.91%** |
| Accuracy gap from baseline | **−5.09%** |

---

## Interactive Dashboard

The project ships with a fully interactive **Streamlit Web Application** for live demonstration.

### Features

- **Digital Canvas** — Draw digits (0–9) directly in-browser
- **Dynamic Corruption** — Inject custom levels of Gaussian noise via a real-time slider
- **Live Inference** — Watch tensors flow through the Healer and Expert in real-time
- **Probability Analysis** — View the CNN's Softmax probability distributions via dynamic bar charts
- **Cached Models** — Models load once and stay in memory for instant inference

### Pipeline Visualization in the Dashboard

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  You Draw a │    │ Noise Added  │    │  VAE Heals   │
│    Digit    │ ─► │ (Adjustable) │ ─► │  the Image   │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  CNN Classifies   │
                                    │ Healed vs Noisy   │
                                    │  (Side-by-Side)   │
                                    └──────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/RushikeshMasalkar/SelfHealingNN.git
cd SelfHealingNN
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy pandas streamlit streamlit-drawable-canvas Pillow
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Notebook Pipeline

The complete training pipeline is broken into **5 sequential Jupyter notebooks**, designed to be run in order:

| # | Notebook | Description |
|:-:|:---|:---|
| 01 | **Data & Noise Pipeline** | Load MNIST, visualize clean vs. corrupted samples, build noise injection utilities |
| 02 | **Healer VAE** | Train the Variational Autoencoder to reconstruct clean images from noisy inputs |
| 03 | **Expert CNN** | Train the Convolutional Neural Network on pristine data as the baseline classifier |
| 04 | **Integration & Testing** | Connect both models into a single pipeline and benchmark against noisy data |
| 05 | **End-to-End Training** | Joint fine-tuning of both networks with the combined loss function |

> All notebooks are in the [`notebooks/`](notebooks/) directory and contain detailed markdown explanations alongside the code.

---

## Repository Structure

```
SelfHealingNN/
│
├── app.py                              # Streamlit web application
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore rules
│
├── models/                             # Trained PyTorch model weights
│   ├── healer_vae.pth                  #   VAE (pre fine-tuning)
│   ├── healer_vae_finetuned.pth        #   VAE (after joint training)
│   ├── expert_cnn.pth                  #   CNN (pre fine-tuning)
│   └── expert_cnn_finetuned.pth        #   CNN (after joint training)
│
├── notebooks/                          # Step-by-step training pipeline
│   ├── 01_Data_and_Noise.ipynb         #   Data loading & noise visualization
│   ├── 02_Healer_VAE.ipynb             #   VAE training
│   ├── 03_Expert_CNN.ipynb             #   CNN training
│   ├── 04_Integration_and_Testing.ipynb#   Pipeline integration & benchmarks
│   └── 05_End_to_End_Training.ipynb    #   Joint end-to-end fine-tuning
│
└── data/                               # MNIST dataset (auto-downloaded)
    └── MNIST/raw/                      #   Raw binary files
```

---

## Tech Stack

<div align="center">

| Category | Technology |
|:---|:---|
| **Deep Learning** | PyTorch, torchvision |
| **Frontend** | Streamlit, streamlit-drawable-canvas |
| **Data Processing** | NumPy, Pandas, Pillow |
| **Notebooks** | Jupyter |
| **Dataset** | MNIST (handwritten digits) |

</div>

---

## Roadmap

The current architecture is a successful **Proof-of-Concept** on MNIST. The next phase targets real-world **Medical Imaging**.

- [ ] **Resolution Scaling** — Upgrade the VAE to handle $128 \times 128$ complex tissue structures
- [ ] **Advanced Backbone** — Replace the Expert CNN with a deep **ResNet** architecture
- [ ] **Domain-Specific Noise** — Simulate real-world medical sensor corruption (Poisson Noise)
- [ ] **Medical Dataset** — Transition to **Pneumonia Chest X-Ray** classification
- [ ] **Model Export** — ONNX export for edge deployment

---

<div align="center">

### Star this repo if you find it useful!

Made by [Rushikesh Masalkar](https://github.com/RushikeshMasalkar)

</div>