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

**Two complete pipelines — MNIST Digit Recognition & Chest X-Ray Pneumonia Detection**

[Explore the Notebooks](#notebook-pipeline) · [Run the Dashboard](#quick-start) · [View Benchmarks](#performance--benchmarks)

</div>

<br/>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Pipeline 1 — MNIST Digit Recognition](#pipeline-1--mnist-digit-recognition)
- [Pipeline 2 — Chest X-Ray Pneumonia Detection](#pipeline-2--chest-x-ray-pneumonia-detection)
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

The architecture has been validated on **two distinct domains**:

1. **MNIST Handwritten Digits** — Proof-of-concept on 28x28 grayscale digits (10 classes)
2. **Chest X-Ray Pneumonia Detection** — Real-world medical imaging on 128x128 X-rays (2 classes: Normal / Pneumonia)

### How It Works

```
┌──────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌────────────────┐
│              │      │   NOISE LAYER   │      │  HEALER (VAE)   │      │  EXPERT (CNN)  │
│  Clean Input │ ───► │  Gaussian Noise │ ───► │   Denoising &   │ ───► │ Classification │
│              │      │  σ = 0.0 – 1.0  │      │  Reconstruction │      │                │
└──────────────┘      └─────────────────┘      └─────────────────┘      └────────────────┘
                                                       │
                                          Reparameterization Trick
                                            z = μ + σ ⊙ ε
```

> **Core Objective:** Prevent model collapse in catastrophically degraded environments by separating noise reduction from classification into two specialized, jointly optimized networks.

---

## Key Results

<div align="center">

### MNIST Pipeline

| Scenario | Accuracy | Change |
|:---|:---:|:---:|
| **Baseline** — CNN on clean data | **98.91%** | — |
| **Disaster** — CNN on noisy data (σ=0.5) | **47.92%** | −51.0% |
| **Self-Healing Pipeline** (fine-tuned) | **93.82%** | **+45.9%** recovery |

### Medical Pipeline (Chest X-Ray)

| Scenario | Accuracy | Change |
|:---|:---:|:---:|
| **Baseline** — CNN on clean X-rays | **87.66%** | — |
| **Disaster** — CNN on noisy X-rays (σ=0.5) | **62.50%** | −25.2% |
| **Self-Healing Pipeline** (fine-tuned) | **80.77%** | **+18.3%** recovery |

</div>

> The Self-Healing pipeline recovered **+45.90%** accuracy on MNIST and **+18.27%** on medical X-rays — demonstrating that the architecture generalizes from simple digits to complex medical imagery.

---

## Pipeline 1 — MNIST Digit Recognition

The first pipeline serves as the **proof-of-concept**, validating the self-healing architecture on the MNIST handwritten digit dataset.

### Dataset

| Property | Value |
|:---|:---|
| **Images** | 70,000 (60K train / 10K test) |
| **Resolution** | 28 x 28 grayscale |
| **Classes** | 10 (digits 0–9) |
| **Source** | PyTorch built-in MNIST |

### Healer — DenoisingVAE

A lightweight Variational Autoencoder designed for 28x28 inputs.

| Layer | Details |
|:---|:---|
| Encoder | `Conv2d(1→32, stride=2)` → `ReLU` → `Conv2d(32→64, stride=2)` → `ReLU` |
| Latent Space | `Linear(3136→20)` for μ, `Linear(3136→20)` for log σ² — **20-dim latent** |
| Decoder | `Linear(20→3136)` → `ConvTranspose2d(64→32)` → `ConvTranspose2d(32→1)` → `Sigmoid` |

### Expert — ExpertCNN

A 2-block CNN classifier trained on clean digits.

| Layer | Details |
|:---|:---|
| Conv Block 1 | `Conv2d(1→32)` → `ReLU` → `MaxPool2d(2×2)` |
| Conv Block 2 | `Conv2d(32→64)` → `ReLU` → `MaxPool2d(2×2)` |
| Classifier | `Linear(3136→128)` → `ReLU` → `Linear(128→10)` |

### Training

| Phase | Epochs | Noise σ | Learning Rate | Details |
|:---|:---:|:---:|:---:|:---|
| **VAE Pre-training** | 5 | 0.4 | 1e-3 | BCE + KLD loss |
| **CNN Pre-training** | 5 | — | 1e-3 | CrossEntropy on clean data |
| **End-to-End** | 3 | 0.5 | 1e-4 | Combined loss, λ=10.0 |

### Results

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

## Pipeline 2 — Chest X-Ray Pneumonia Detection

The second pipeline scales the architecture to **real-world medical imaging**, tackling binary classification of chest X-rays into Normal vs. Pneumonia.

### Dataset

| Property | Value |
|:---|:---|
| **Images** | 5,856 total (5,216 train / 624 test / 16 val) |
| **Resolution** | 128 x 128 grayscale (resized from variable originals) |
| **Classes** | 2 (NORMAL / PNEUMONIA) |
| **Class Distribution** | NORMAL: 1,341 — PNEUMONIA: 3,875 (~1:2.9 imbalance) |
| **Source** | Chest X-Ray Pneumonia dataset (Kaggle) |

### Class Imbalance Handling

The dataset is heavily skewed toward PNEUMONIA. Two strategies address this:

1. **WeightedRandomSampler** — Oversamples NORMAL during training to balance each batch
2. **Weighted CrossEntropyLoss** — NORMAL class weight = 2.89, PNEUMONIA = 1.0

### Healer — DeepMedicalVAE

A deep 4-layer VAE with BatchNorm, designed for 128x128 medical images. **7,084,929 parameters.**

| Layer | Details |
|:---|:---|
| Encoder Block 1 | `Conv2d(1→32, stride=2)` → `BatchNorm2d` → `ReLU` |
| Encoder Block 2 | `Conv2d(32→64, stride=2)` → `BatchNorm2d` → `ReLU` |
| Encoder Block 3 | `Conv2d(64→128, stride=2)` → `BatchNorm2d` → `ReLU` |
| Encoder Block 4 | `Conv2d(128→256, stride=2)` → `BatchNorm2d` → `ReLU` |
| Latent Space | `Linear(16384→128)` for μ, `Linear(16384→128)` for log σ² — **128-dim latent** |
| Decoder | 4× `ConvTranspose2d` with `BatchNorm2d` → `Sigmoid` output |

### Expert — MedicalExpertCNN

A deep 4-block CNN with BatchNorm and Dropout. **8,778,946 parameters.**

| Layer | Details |
|:---|:---|
| Conv Block 1 | `Conv2d(1→32)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` |
| Conv Block 2 | `Conv2d(32→64)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` |
| Conv Block 3 | `Conv2d(64→128)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` |
| Conv Block 4 | `Conv2d(128→256)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)` |
| Classifier | `Linear(16384→512)` → `ReLU` → `Dropout(0.5)` → `Linear(512→2)` |

### Training

| Phase | Epochs | Noise σ | Learning Rate | Details |
|:---|:---:|:---:|:---:|:---|
| **VAE Pre-training** | 20 | 0.4 | 1e-3 | BCE + 0.5×KLD, ReduceLROnPlateau, grad clip=1.0 |
| **CNN Pre-training** | 25 | — | 5e-4 | Weighted CrossEntropy, weight_decay=1e-4 |
| **End-to-End** | 15 | 0.4 | 1e-4 | Combined loss, λ=5.0 |

### Data Augmentation

```python
RandomHorizontalFlip()
RandomRotation(10)
RandomAffine(degrees=0, translate=(0.05, 0.05))
```

### Results

```
  88% ┤ ████                                          Baseline (Clean): 87.66%
      ┤ ████
  81% ┤ ████                           ████           Self-Healing:     80.77%
      ┤ ████                           ████
      ┤ ████                           ████
      ┤ ████            ████           ████
  63% ┤ ████            ████           ████            CNN + Noise:      62.50%
      ┤ ████            ████           ████
      ┼─────────────────────────────────────
        Baseline        Disaster       Self-Healing
       (Clean Data)   (CNN + Noise)    (Fine-Tuned)
```

| Metric | Value |
|:---|:---:|
| Accuracy recovered from noise collapse | **+18.27%** |
| Final pipeline accuracy (noisy input) | **80.77%** |
| Baseline accuracy (clean input) | **87.66%** |
| Accuracy gap from baseline | **−6.89%** |

### Per-Class Breakdown (Healed, σ=0.5)

| Class | Correct | Total | Accuracy |
|:---|:---:|:---:|:---:|
| NORMAL | 137 | 234 | 58.5% |
| PNEUMONIA | 368 | 390 | 94.4% |
| **Overall** | **504** | **624** | **80.77%** |

---

## System Architecture

The pipeline consists of two independent neural networks trained sequentially and later **fine-tuned end-to-end** to achieve joint optimization. The same architecture pattern is used for both MNIST and Medical pipelines, scaled appropriately.

### Stage A — The "Healer" (Variational Autoencoder)

A **generative probabilistic model** responsible for strict denoising.

| Component | MNIST | Medical |
|:---|:---|:---|
| **Input** | 1×28×28 | 1×128×128 |
| **Encoder Depth** | 2 Conv layers | 4 Conv layers + BatchNorm |
| **Latent Dim** | 20 | 128 |
| **Decoder Depth** | 2 ConvTranspose layers | 4 ConvTranspose layers + BatchNorm |
| **Parameters** | ~230K | 7,084,929 |

**Reparameterization Trick:**

$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$

**Loss Function:** Dual-objective combining **Binary Cross-Entropy** (reconstruction) and **KL-Divergence** (regularization):

$$\mathcal{L}_{VAE} = \text{BCE}(x, \hat{x}) + \beta \cdot D_{KL}\big(\mathcal{N}(\mu, \sigma^2) \;\|\; \mathcal{N}(0, 1)\big)$$

where β=1.0 for MNIST and β=0.5 for medical images.

### Stage B — The "Expert" (Deep CNN)

A **Convolutional Neural Network** trained strictly on pristine, uncorrupted data. It acts as the baseline classifier, protected from real-world noise by the Healer's preprocessing.

| Component | MNIST | Medical |
|:---|:---|:---|
| **Conv Blocks** | 2 | 4 (with BatchNorm) |
| **Regularization** | None | Dropout(0.5) + weight_decay |
| **Output Classes** | 10 | 2 |
| **Class Balancing** | None | Weighted loss + balanced sampling |
| **Parameters** | ~400K | 8,778,946 |

### End-to-End Joint Optimization

The true power is unlocked by combining both computational graphs into a **single differentiable pipeline**:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{VAE}} + \lambda \cdot \mathcal{L}_{\text{CNN}}$$

where λ=10.0 for MNIST and λ=5.0 for medical images.

This forces the Healer to preserve the **exact geometric features** the Expert requires for classification — not just produce visually clean images.

---

## Performance & Benchmarks

### Side-by-Side Comparison

<div align="center">

| Metric | MNIST | Medical X-Ray |
|:---|:---:|:---:|
| **Clean Accuracy** | 98.91% | 87.66% |
| **Noisy Accuracy** | 47.92% | 62.50% |
| **Healed Accuracy** | 93.82% | 80.77% |
| **Recovery** | **+45.90%** | **+18.27%** |
| **Gap from Baseline** | −5.09% | −6.89% |
| **Image Size** | 28×28 | 128×128 |
| **Total Parameters** | ~630K | 15.9M |

</div>

---

## Interactive Dashboard

The project ships with a **MedHeal AI** Streamlit dashboard — a professional medical imaging interface for the Chest X-Ray pipeline.

### Features

- **X-Ray Upload** — Upload your own chest X-ray or sample randomly from the test set
- **Dynamic Corruption** — Inject custom levels of Gaussian noise via a real-time slider (σ = 0.0–1.0)
- **3-Stage Pipeline View** — Visualize Input → Corrupted → Healed side-by-side with predictions
- **Confidence Analysis** — Custom confidence bars showing NORMAL/PNEUMONIA probabilities at each stage
- **Impact Metrics** — Summary cards showing clean, noisy, and healed confidence with recovery percentages
- **Batch Analysis** — Run 8 random samples at once and see per-sample correctness
- **Full Test Set Evaluation** — Run all 624 test images with per-class accuracy breakdown and progress bar
- **Benchmark Sidebar** — Architecture details, training stats, and benchmark table always visible

### Dashboard Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Upload or   │    │ Noise Added  │    │  VAE Heals   │
│ Sample X-Ray │ ─► │ (Adjustable) │ ─► │  the X-Ray   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                                  ┌──────────────────────┐
                                  │   CNN Classifies:     │
                                  │  NORMAL / PNEUMONIA   │
                                  │                       │
                                  │  Confidence Bars +    │
                                  │  Impact Summary       │
                                  └──────────────────────┘
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
pip install torch torchvision numpy pandas streamlit Pillow
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### 5. Run the Training Notebooks (Optional)

To retrain from scratch, run the notebooks in `notebooks/` sequentially:
- Notebooks **01–05** for the MNIST pipeline
- Notebooks **06–10** for the Medical pipeline

---

## Notebook Pipeline

The complete training pipeline is broken into **10 sequential Jupyter notebooks** across two domains:

### MNIST Pipeline (Notebooks 01–05)

| # | Notebook | Description |
|:-:|:---|:---|
| 01 | **Data & Noise Pipeline** | Load MNIST, visualize clean vs. corrupted samples, build noise injection utilities |
| 02 | **Healer VAE** | Train the DenoisingVAE (2-layer encoder/decoder, 20-dim latent, 5 epochs) |
| 03 | **Expert CNN** | Train the ExpertCNN on pristine data (2 conv blocks, 10 classes) |
| 04 | **Integration & Testing** | Connect both models into a single pipeline and benchmark against noisy data |
| 05 | **End-to-End Training** | Joint fine-tuning with combined loss (λ=10.0, 3 epochs, lr=1e-4) |

### Medical Pipeline (Notebooks 06–10)

| # | Notebook | Description |
|:-:|:---|:---|
| 06 | **Medical Data Pipeline** | Load Chest X-Ray dataset, visualize samples, set up transforms, balanced sampling, and noise utilities |
| 07 | **Medical Healer VAE** | Train the DeepMedicalVAE (4-layer + BatchNorm, 128-dim latent, 7.1M params, 20 epochs) |
| 08 | **Medical Expert CNN** | Train the MedicalExpertCNN (4-layer + BatchNorm + Dropout, 8.8M params, 25 epochs, 87.18% test accuracy) |
| 09 | **Medical Pipeline Test** | Full pipeline test — measure clean, noisy, and healed accuracy with per-class breakdown |
| 10 | **End-to-End Medical** | Joint fine-tuning with combined loss (λ=5.0, 15 epochs, lr=1e-4, 80.77% healed accuracy) |

> All notebooks are in the [`notebooks/`](notebooks/) directory and contain detailed markdown explanations alongside the code.

---

## Repository Structure

```
SelfHealingNN/
│
├── app.py                                # Streamlit medical dashboard (MedHeal AI)
├── README.md                             # Project documentation
├── .gitignore                            # Git ignore rules
│
├── models/                               # Trained PyTorch model weights (.pth)
│   ├── healer_vae.pth                    #   MNIST VAE (pre fine-tuning)
│   ├── healer_vae_finetuned.pth          #   MNIST VAE (after end-to-end)
│   ├── expert_cnn.pth                    #   MNIST CNN (pre fine-tuning)
│   ├── expert_cnn_finetuned.pth          #   MNIST CNN (after end-to-end)
│   ├── medical_healer.pth                #   Medical VAE (pre fine-tuning)
│   ├── medical_healer_finetuned.pth      #   Medical VAE (after end-to-end)
│   ├── medical_expert.pth                #   Medical CNN (pre fine-tuning)
│   └── medical_expert_finetuned.pth      #   Medical CNN (after end-to-end)
│
├── notebooks/                            # Step-by-step training pipeline
│   ├── 01_Data_and_Noise.ipynb           #   MNIST data loading & noise visualization
│   ├── 02_Healer_VAE.ipynb               #   MNIST VAE training
│   ├── 03_Expert_CNN.ipynb               #   MNIST CNN training
│   ├── 04_Integration_and_Testing.ipynb  #   MNIST pipeline integration & benchmarks
│   ├── 05_End_to_End_Training.ipynb      #   MNIST joint fine-tuning
│   ├── 06_Medical_Data_Pipeline.ipynb    #   X-Ray data loading & preprocessing
│   ├── 07_Medical_Healer.ipynb           #   Medical VAE training (7.1M params)
│   ├── 08_Medical_Expert_CNN.ipynb       #   Medical CNN training (8.8M params)
│   ├── 09_Medical_Pipeline_Test.ipynb    #   Medical pipeline benchmarks
│   └── 10_End_to_End_Medical.ipynb       #   Medical joint fine-tuning
│
├── data/                                 # MNIST dataset (auto-downloaded)
│   └── MNIST/raw/                        #   Raw binary files
│
└── medical_data/                         # Chest X-Ray Pneumonia dataset
    ├── train/                            #   5,216 images
    │   ├── NORMAL/                       #     1,341 images
    │   └── PNEUMONIA/                    #     3,875 images
    ├── test/                             #   624 images
    │   ├── NORMAL/                       #     234 images
    │   └── PNEUMONIA/                    #     390 images
    └── val/                              #   16 images
        ├── NORMAL/                       #     8 images
        └── PNEUMONIA/                    #     8 images
```

---

## Tech Stack

<div align="center">

| Category | Technology |
|:---|:---|
| **Deep Learning** | PyTorch, torchvision |
| **Models** | Variational Autoencoder (VAE), Convolutional Neural Network (CNN) |
| **Frontend** | Streamlit |
| **Data Processing** | NumPy, Pandas, Pillow |
| **Notebooks** | Jupyter |
| **Datasets** | MNIST (handwritten digits), Chest X-Ray Pneumonia |

</div>

---

## Roadmap

The architecture has been validated on two domains. Future work targets broader applications and stronger models.

- [x] **MNIST Proof-of-Concept** — Validate self-healing architecture on handwritten digits
- [x] **Resolution Scaling** — Upgrade the VAE to handle 128x128 complex tissue structures
- [x] **Medical Dataset** — Transition to Pneumonia Chest X-Ray classification
- [x] **Deep Architectures** — 4-layer encoder/decoder with BatchNorm for medical images
- [x] **Class Imbalance Handling** — WeightedRandomSampler + weighted loss
- [x] **Medical Dashboard** — Professional Streamlit UI for X-Ray analysis
- [ ] **Additional Datasets** — Extend to more medical imaging tasks (brain MRI, skin lesion, retinal scans)
- [ ] **Advanced Backbone** — Replace Expert CNN with ResNet / EfficientNet
- [ ] **Domain-Specific Noise** — Simulate real-world medical sensor corruption (Poisson noise, motion blur)
- [ ] **Model Export** — ONNX export for edge deployment
- [ ] **Adversarial Robustness** — Test self-healing against adversarial attacks

---

<div align="center">

### Star this repo if you find it useful!

Made by [Rushikesh Masalkar](https://github.com/RushikeshMasalkar)

</div>
