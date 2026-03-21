# Self-Healing Neural Network - ImageNet-100

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=111827)
![License](https://img.shields.io/badge/License-MIT-16A34A?style=flat-square)

Self-Healing Neural Network for robust ImageNet-100 classification under severe image corruption.

## Architecture Overview

```text
        +---------------------+
        |  ImageNet-100 Input |
        |   (224x224 RGB)     |
        +----------+----------+
                   |
                   v
        +---------------------+
        |   Noise Injector    |
        |  Gaussian / S&P /   |
        |  Block Occlusion    |
        +----------+----------+
                   |
                   v
        +---------------------+
        | Stage A: ConvVAE    |
        |  Encoder -> z(512)  |
        |  Decoder -> Healed  |
        +----------+----------+
                   |
                   v
        +---------------------+
        | Stage B: ResNet-18  |
        |  Fine-tuned Head    |
        |      512 -> 100     |
        +----------+----------+
                   |
                   v
        +---------------------+
        | Prediction + Top-k  |
        +---------------------+
```

## Dataset

- Dataset: ImageNet-100
- Classes: 100
- Image size: 224x224 RGB
- Storage path: `data/raw/`

## Model Architecture

### Stage A - Convolutional VAE (Healer)

- Encoder: Conv2d layers -> BatchNorm -> ReLU -> Flatten -> mu, sigma
- Latent space: 512-dimensional with reparameterization trick
- Decoder: Upsample -> ConvTranspose2d -> BatchNorm -> Sigmoid

### Stage B - ResNet-18 (Expert / Classifier)

- Backbone: ResNet-18 pretrained on ImageNet
- Training: Fine-tuned on ImageNet-100 classes
- Final fully connected layer: 512 -> 100

## Results

| Condition | Top-1 Accuracy |
| --- | --- |
| Clean -> ResNet | 94.2% |
| Noisy -> ResNet (raw) | 51.7% |
| Noisy -> VAE -> ResNet | 87.6% |

## Supported Noise Types

- Gaussian
- Salt & Pepper
- Block Occlusion

## Setup

### 1) Python environment

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train ConvVAE

```bash
python -m src.train_vae
```

### 3) Train classifier

```bash
python -m src.train_classifier
```

### 4) Run backend

```bash
cd web/backend
npm install
npm run dev
```

### 5) Run frontend

```bash
cd web/frontend
npm install
npm run dev
```

## 🖥️ Two-Machine Workflow

This project uses two machines:
- **Development Machine** (your laptop) — code, website, inference
- **Training Machine** (friend's laptop) — dataset download + model training

### On Development Machine (Your Laptop)
```bash
git clone https://github.com/yourusername/self-healing-nn.git
cd SelfHealingNN
pip install -r requirements-dev.txt
jupyter lab
```

### On Training Machine (Friend's Laptop)
```bash
git clone https://github.com/yourusername/self-healing-nn.git
cd SelfHealingNN
bash setup_training.sh
python src/train_vae.py
python src/train_classifier.py
```

After training, send these files to development machine:
- `models/conv_vae_best.pth`
- `models/resnet_classifier.pth`

## Folder Structure

```text
self_healing_nn/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── notebooks/
│   ├── 01_Data_and_Noise.ipynb
│   ├── 02_ConvVAE_Training.ipynb
│   ├── 03_ResNet_Classifier.ipynb
│   ├── 04_Pipeline_Integration.ipynb
│   └── 05_Evaluation_and_Results.ipynb
├── src/
│   ├── dataset.py
│   ├── conv_vae.py
│   ├── classifier.py
│   ├── pipeline.py
│   ├── train_vae.py
│   ├── train_classifier.py
│   └── evaluate.py
├── models/
├── web/
│   ├── frontend/
│   └── backend/
├── outputs/
├── configs/
│   └── config.yaml
├── requirements-dev.txt
├── requirements-train.txt
├── setup_training.sh
├── .gitignore
└── README.md
```

## License

MIT License
