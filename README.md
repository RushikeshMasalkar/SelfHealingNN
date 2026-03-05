<div align="center">
<h1>Self-Healing Neural Network Architecture</h1>
<p><b>A Two-Stage Deep Learning Pipeline for Catastrophic Noise Recovery & Classification</b></p>

</div>

Project Overview

Standard image classification models degrade rapidly when exposed to noisy, real-world data. This project introduces a "Self-Healing" AI pipeline. Instead of feeding corrupted data directly into a classifier, this architecture intercepts the data, mathematically filters out the noise using a generative model, and passes the reconstructed data to the classifier.

This repository contains the complete PyTorch backend, training notebooks, and an interactive Streamlit Web Application to visualize the pipeline in real-time.

System Architecture

The pipeline consists of two independent neural networks trained sequentially, and later fine-tuned simultaneously.

1. Stage A: The "Healer" (Variational Autoencoder - VAE)

A generative probabilistic model responsible for strict denoising.

Encoder: Compresses the noisy input spatially using Conv2d layers.

Latent Space (Reparameterization Trick): Maps features to a continuous distribution ($z = \mu + \sigma \odot \epsilon$), forcing the network to discard random Gaussian static and retain core structural features.

Decoder: Upsamples the latent vector via ConvTranspose2d back to a clean image space.

Loss Function: Optimized using Binary Cross-Entropy (Reconstruction) + KL-Divergence (Regularization).

2. Stage B: The "Expert" (Deep CNN)

A standard Convolutional Neural Network trained strictly on pristine, uncorrupted data. It acts as the baseline classifier, completely protected from real-world noise by the Healer.

Performance & Benchmarks

The true power of this architecture was unlocked during End-to-End Fine-Tuning. By combining the computational graphs of both models ($\mathcal{L}_{Total} = \mathcal{L}_{VAE} + \lambda \cdot \mathcal{L}_{CNN}$), the Healer was forced to preserve the exact geometric features the Expert required for classification.

Testing on Catastrophic Gaussian Noise:
| Model Configuration | Accuracy | Status |
| :--- | :---: | :--- |
| Baseline (Clean Data) | 98.91% | Ideal Conditions |
| Disaster (CNN Only + Noisy Data) | 47.92% | Model Collapse |
| Self-Healing Pipeline (Fine-Tuned) | 93.82% | Recovered |

Result: The Self-Healing pipeline recovered +45.90% accuracy compared to a standalone CNN under heavy noise conditions.

Interactive Web Dashboard

This project includes a fully interactive UI built with Streamlit.

Digital Canvas: Draw digits manually.

Dynamic Corruption: Inject custom levels of static noise via a UI slider.

Live Inference: Watch the tensors pass through the Healer and Expert in real-time.

Probability Analysis: View the CNN's exact Softmax probability distributions via interactive bar charts.

How to Run Locally

Clone the repository:

git clone [https://github.com/RushikeshMasalkar/SelfHealingNN.git](https://github.com/YOUR_USERNAME/SelfHealingNN.git)
cd SelfHealingNN



Activate the Virtual Environment (Windows):

.\venv\Scripts\activate



Install Dependencies:

pip install torch torchvision numpy pandas streamlit streamlit-drawable-canvas Pillow



Launch the Dashboard:

streamlit run app.py



Project Structure

SelfHealingNN/
│
├── data/                       # Dataset directory (MNIST)
├── models/                     # Saved PyTorch weights (.pth)
│   ├── healer_vae.pth
│   └── expert_cnn.pth
│
├── notebooks/                  # Step-by-step training pipeline
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Healer_VAE.ipynb
│   ├── 03_Expert_CNN.ipynb
│   ├── 04_Integration_and_Testing.ipynb
│   └── 05_End_to_End_Training.ipynb
│
├── app.py                      # Streamlit Web Application
├── .gitignore
└── README.md



Future Scope (Phase 2)

The current architecture serves as a highly successful Proof-of-Concept on the MNIST dataset. The upcoming phase will transition this architecture to Medical Imaging (Pneumonia Chest X-Rays).

Upgrading the VAE to handle $128 \times 128$ complex structures.

Replacing the Expert CNN with a deep ResNet architecture.

Simulating real-world medical sensor corruption (Poisson Noise) to build a robust diagnostic firewall.