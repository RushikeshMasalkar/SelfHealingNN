<div align="center">

Self-Healing Neural Network Architecture

A Two-Stage Deep Learning Pipeline for Catastrophic Noise Recovery & Classification

</div>

<hr>

<details>
<summary><b>Table of Contents (Click to Expand)</b></summary>
<ol>
<li><a href="#abstract--project-overview">Abstract & Project Overview</a></li>
<li><a href="#system-architecture--mathematical-foundation">System Architecture</a></li>
<li><a href="#performance--benchmarks">Performance Benchmarks</a></li>
<li><a href="#interactive-web-dashboard">Interactive Web Dashboard</a></li>
<li><a href="#installation--deployment">Installation & Deployment</a></li>
<li><a href="#repository-structure">Repository Structure</a></li>
<li><a href="#future-scope-phase-2">Future Scope</a></li>
</ol>
</details>

Abstract & Project Overview

Standard image classification models degrade rapidly when exposed to noisy, real-world data. This project introduces a Self-Healing AI pipeline. Instead of feeding corrupted data directly into a classifier, this architecture intercepts the data, mathematically filters out the noise using a generative model, and passes the reconstructed data to the classifier.

The Core Objective: Mathematically isolate the noise-reduction task from the classification task, preventing model collapse in catastrophically degraded environments.

This repository contains the complete PyTorch backend, training notebooks, and an interactive Streamlit Web Application to visualize the pipeline in real-time.

System Architecture & Mathematical Foundation

The pipeline consists of two independent neural networks trained sequentially, and later fine-tuned simultaneously to achieve end-to-end optimization.

Stage A: The "Healer" (Variational Autoencoder)

A generative probabilistic model responsible for strict denoising.

Encoder: Compresses the noisy input spatially using sequential Conv2d layers.

Latent Space (Reparameterization Trick): Maps features to a continuous distribution, forcing the network to discard random Gaussian static and retain core structural features.






$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, 1)$$



Decoder: Upsamples the latent vector via ConvTranspose2d back to a clean image space.

Optimization: Trained using a dual-objective loss function combining Binary Cross-Entropy (Reconstruction) and KL-Divergence (Regularization).

Stage B: The "Expert" (Deep CNN)

A Convolutional Neural Network trained strictly on pristine, uncorrupted data. It acts as the baseline classifier, protected from real-world noise by the Healer's preprocessing.

Performance & Benchmarks

The true power of this architecture was unlocked during End-to-End Fine-Tuning. By combining the computational graphs of both models, the Healer was forced to preserve the exact geometric features the Expert required for classification.

Joint Optimization Function:


$$\mathcal{L}_{Total} = \mathcal{L}_{VAE} + \lambda \cdot \mathcal{L}_{CNN}$$

Testing on Catastrophic Gaussian Noise:

Model Configuration

Accuracy

Network State

Baseline (Clean Data)

98.91%

Optimal Conditions

Disaster (CNN Only + Noisy Data)

47.92%

Model Collapse

Self-Healing Pipeline (Fine-Tuned)

93.82%

Recovered

Result: The Self-Healing pipeline recovered +45.90% accuracy compared to a standalone CNN under heavy noise conditions.

Interactive Web Dashboard

This project includes a fully interactive UI built with Streamlit for live demonstration.

Digital Canvas: Draw input features manually.

Dynamic Corruption: Inject custom levels of static noise via a UI slider.

Live Inference: Watch the tensors pass through the Healer and Expert in real-time.

Probability Analysis: View the CNN's exact Softmax probability distributions via dynamic charting.

Installation & Deployment

Follow these steps to replicate the environment and run the dashboard locally.

1. Clone the repository

git clone [https://github.com/RushikeshMasalkar/SelfHealingNN.git](https://github.com/YOUR_USERNAME/SelfHealingNN.git)
cd SelfHealingNN


2. Activate the Virtual Environment (Windows)

python -m venv venv
.\venv\Scripts\activate


3. Install Dependencies

pip install torch torchvision numpy pandas streamlit streamlit-drawable-canvas Pillow


4. Launch the Dashboard

streamlit run app.py


Repository Structure

SelfHealingNN/
├── data/                       # Dataset directory
├── models/                     # Saved PyTorch weights (.pth)
│   ├── healer_vae.pth
│   └── expert_cnn.pth
├── notebooks/                  # Step-by-step training pipeline
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Healer_VAE.ipynb
│   ├── 03_Expert_CNN.ipynb
│   ├── 04_Integration_and_Testing.ipynb
│   └── 05_End_to_End_Training.ipynb
├── app.py                      # Streamlit Web Application
├── .gitignore
└── README.md                   # Project Documentation


Future Scope (Phase 2)

The current architecture serves as a highly successful Proof-of-Concept on the MNIST dataset. The upcoming phase will transition this architecture to Medical Imaging (Pneumonia Chest X-Rays).

Resolution Scaling: Upgrading the VAE to handle $128 \times 128$ complex bone and tissue structures.

Advanced Architecture: Replacing the Expert CNN with a deep ResNet architecture.

Domain-Specific Noise: Simulating real-world medical sensor corruption (Poisson Noise) to build a robust diagnostic firewall.