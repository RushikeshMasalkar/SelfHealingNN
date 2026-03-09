import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedHeal AI - Medical Self-Healing Pipeline",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }

    /* Header */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1.2;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .metric-delta-up {
        color: #22c55e;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-delta-down {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Status badges */
    .badge-normal {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .badge-pneumonia {
        display: inline-block;
        background: #7f1d1d;
        color: #fca5a5;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    /* Pipeline step labels */
    .step-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 8px;
    }
    .step-label-active {
        color: #6366f1;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* Confidence bar */
    .confidence-bar-bg {
        background: #1e293b;
        border-radius: 8px;
        height: 28px;
        width: 100%;
        overflow: hidden;
        border: 1px solid #334155;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.75rem;
        color: white;
        transition: width 0.6s ease;
    }

    /* Image containers */
    .img-container {
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 8px;
        background: #0f172a;
        text-align: center;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
class DeepMedicalVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(DeepMedicalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x).view(-1, 256 * 8 * 8)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h2 = F.relu(self.fc_decode(z)).view(-1, 256, 8, 8)
        return self.decoder(h2), mu, logvar


class MedicalExpertCNN(nn.Module):
    def __init__(self):
        super(MedicalExpertCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_medical_models():
    healer = DeepMedicalVAE()
    expert = MedicalExpertCNN()
    healer.load_state_dict(torch.load(
        './models/medical_healer_finetuned.pth',
        map_location=torch.device('cpu'), weights_only=True
    ))
    expert.load_state_dict(torch.load(
        './models/medical_expert_finetuned.pth',
        map_location=torch.device('cpu'), weights_only=True
    ))
    healer.eval()
    expert.eval()
    return healer, expert


@st.cache_resource
def load_test_dataset():
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(
        root='./medical_data/test',
        transform=test_transforms
    )
    return dataset


healer, expert = load_medical_models()
test_dataset = load_test_dataset()

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
IMAGE_SIZE = 128


# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def add_noise(tensor, factor):
    noise = torch.randn_like(tensor) * factor
    return torch.clamp(tensor + noise, 0., 1.)


def run_pipeline(image_tensor, noise_factor):
    """Run full healing pipeline and return all results."""
    with torch.no_grad():
        noisy = add_noise(image_tensor, noise_factor)
        healed, mu, logvar = healer(noisy)

        logits_clean = expert(image_tensor)
        probs_clean = F.softmax(logits_clean, dim=1).squeeze().numpy()
        pred_clean = int(np.argmax(probs_clean))

        logits_noisy = expert(noisy)
        probs_noisy = F.softmax(logits_noisy, dim=1).squeeze().numpy()
        pred_noisy = int(np.argmax(probs_noisy))

        logits_healed = expert(healed)
        probs_healed = F.softmax(logits_healed, dim=1).squeeze().numpy()
        pred_healed = int(np.argmax(probs_healed))

    return {
        'noisy': noisy,
        'healed': healed,
        'pred_clean': pred_clean, 'probs_clean': probs_clean,
        'pred_noisy': pred_noisy, 'probs_noisy': probs_noisy,
        'pred_healed': pred_healed, 'probs_healed': probs_healed,
    }


def render_confidence_bar(label, confidence, color):
    pct = confidence * 100
    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
            <span style="font-size:0.8rem; color:#cbd5e1; font-weight:500;">{label}</span>
            <span style="font-size:0.8rem; color:#e2e8f0; font-weight:600;">{pct:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}cc);"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_badge(pred_class):
    if pred_class == 0:
        return '<span class="badge-normal">NORMAL</span>'
    else:
        return '<span class="badge-pneumonia">PNEUMONIA</span>'


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">MedHeal AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Self-Healing Neural Network for Chest X-Ray Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")

    input_mode = st.radio(
        "Input Source",
        ["Upload X-Ray", "Random from Test Set"],
        index=1,
        help="Upload your own chest X-ray or randomly sample from the test set."
    )

    noise_factor = st.slider(
        "Noise Intensity",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Simulates sensor degradation, motion blur, or low-dose artifacts."
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("## Architecture")
    st.markdown("""
    **Healer** — Deep VAE
    4-layer encoder/decoder, BatchNorm
    128-dim latent space, 7.1M params

    **Expert** — Deep CNN
    4-layer feature extractor, BatchNorm
    Dropout regularization, 8.8M params

    **Training**
    End-to-end fine-tuned jointly
    Weighted loss for class imbalance
    """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("## Benchmark")
    st.markdown("""
    | Scenario | Accuracy |
    |:---|:---:|
    | Clean X-rays | 87.66% |
    | Noisy X-rays | 62.50% |
    | **Healed** | **80.77%** |
    | Recovery | **+18.27%** |
    """)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("## Dataset")
    st.markdown("""
    Chest X-Ray Pneumonia
    5,216 train / 624 test images
    Classes: Normal, Pneumonia
    """)


# ─────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────
image_tensor = None
ground_truth = None

if input_mode == "Upload X-Ray":
    uploaded = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Grayscale or color X-ray image. Will be converted to 128x128 grayscale."
    )
    if uploaded is not None:
        image_tensor = preprocess_uploaded_image(uploaded)
else:
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        random_btn = st.button("Sample Random", type="primary", use_container_width=True)
    with col_btn2:
        batch_btn = st.button("Run Batch (8)", use_container_width=True)

    if random_btn:
        idx = random.randint(0, len(test_dataset) - 1)
        image_tensor, ground_truth = test_dataset[idx]
        image_tensor = image_tensor.unsqueeze(0)
        st.session_state['current_image'] = image_tensor
        st.session_state['ground_truth'] = ground_truth
    elif batch_btn:
        st.session_state['batch_mode'] = True
    elif 'current_image' in st.session_state:
        image_tensor = st.session_state['current_image']
        ground_truth = st.session_state.get('ground_truth')

# ─────────────────────────────────────────────────────────────
# BATCH MODE
# ─────────────────────────────────────────────────────────────
if st.session_state.get('batch_mode'):
    st.session_state['batch_mode'] = False
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Batch Analysis — 8 Random Test Samples")

    indices = random.sample(range(len(test_dataset)), 8)
    correct_clean, correct_noisy, correct_healed = 0, 0, 0

    cols = st.columns(4)
    for i, idx in enumerate(indices):
        img_t, gt = test_dataset[idx]
        img_t = img_t.unsqueeze(0)
        result = run_pipeline(img_t, noise_factor)

        if result['pred_clean'] == gt:
            correct_clean += 1
        if result['pred_noisy'] == gt:
            correct_noisy += 1
        if result['pred_healed'] == gt:
            correct_healed += 1

        with cols[i % 4]:
            gt_label = CLASS_NAMES[gt]
            healed_label = CLASS_NAMES[result['pred_healed']]
            match = result['pred_healed'] == gt

            st.image(
                img_t.squeeze().numpy(),
                caption=f"GT: {gt_label}",
                use_container_width=True,
            )
            st.image(
                result['healed'].squeeze().numpy(),
                caption=f"Healed → {healed_label}",
                use_container_width=True,
            )
            if match:
                st.success("Correct", icon="✓")
            else:
                st.error("Incorrect", icon="✗")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Clean Accuracy", f"{correct_clean}/8")
    with mc2:
        st.metric("Noisy Accuracy", f"{correct_noisy}/8")
    with mc3:
        st.metric("Healed Accuracy", f"{correct_healed}/8")
    with mc4:
        recovery = correct_healed - correct_noisy
        st.metric("Recovery", f"+{recovery}" if recovery >= 0 else str(recovery))


# ─────────────────────────────────────────────────────────────
# SINGLE IMAGE PIPELINE
# ─────────────────────────────────────────────────────────────
if image_tensor is not None and not st.session_state.get('batch_mode'):
    result = run_pipeline(image_tensor, noise_factor)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Pipeline Visualization ──
    st.markdown("### Pipeline Stages")
    p1, arrow1, p2, arrow2, p3 = st.columns([3, 0.5, 3, 0.5, 3])

    with p1:
        st.markdown('<p class="step-label">STEP 1 — INPUT</p>', unsafe_allow_html=True)
        st.image(image_tensor.squeeze().numpy(), use_container_width=True, clamp=True)
        if ground_truth is not None:
            st.markdown(f"Ground Truth: {render_prediction_badge(ground_truth)}", unsafe_allow_html=True)

    with arrow1:
        st.markdown("<div style='display:flex;align-items:center;height:200px;justify-content:center;'>"
                     "<span style='font-size:2rem;color:#475569;'>→</span></div>",
                     unsafe_allow_html=True)

    with p2:
        st.markdown(f'<p class="step-label">STEP 2 — CORRUPTED (σ={noise_factor:.2f})</p>', unsafe_allow_html=True)
        st.image(result['noisy'].squeeze().numpy(), use_container_width=True, clamp=True)
        st.markdown(f"Prediction: {render_prediction_badge(result['pred_noisy'])}", unsafe_allow_html=True)

    with arrow2:
        st.markdown("<div style='display:flex;align-items:center;height:200px;justify-content:center;'>"
                     "<span style='font-size:2rem;color:#475569;'>→</span></div>",
                     unsafe_allow_html=True)

    with p3:
        st.markdown('<p class="step-label step-label-active">STEP 3 — HEALED (VAE)</p>', unsafe_allow_html=True)
        st.image(result['healed'].squeeze().numpy(), use_container_width=True, clamp=True)
        st.markdown(f"Prediction: {render_prediction_badge(result['pred_healed'])}", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Diagnostic Metrics ──
    st.markdown("### Diagnostic Confidence")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("**Clean Input**")
        render_confidence_bar("NORMAL", result['probs_clean'][0], "#22c55e")
        render_confidence_bar("PNEUMONIA", result['probs_clean'][1], "#ef4444")

    with d2:
        st.markdown("**Corrupted Input**")
        render_confidence_bar("NORMAL", result['probs_noisy'][0], "#22c55e")
        render_confidence_bar("PNEUMONIA", result['probs_noisy'][1], "#ef4444")

    with d3:
        st.markdown("**Healed (VAE)**")
        render_confidence_bar("NORMAL", result['probs_healed'][0], "#22c55e")
        render_confidence_bar("PNEUMONIA", result['probs_healed'][1], "#ef4444")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Summary Cards ──
    st.markdown("### Impact Summary")
    s1, s2, s3, s4 = st.columns(4)

    clean_conf = float(result['probs_clean'].max()) * 100
    noisy_conf = float(result['probs_noisy'].max()) * 100
    healed_conf = float(result['probs_healed'].max()) * 100
    recovery_pct = healed_conf - noisy_conf

    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{clean_conf:.1f}%</div>
            <div class="metric-label">Clean Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{noisy_conf:.1f}%</div>
            <div class="metric-label">Noisy Confidence</div>
            <div class="metric-delta-down">▼ {clean_conf - noisy_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{healed_conf:.1f}%</div>
            <div class="metric-label">Healed Confidence</div>
            <div class="metric-delta-up">▲ {recovery_pct:.1f}% recovered</div>
        </div>
        """, unsafe_allow_html=True)

    with s4:
        healed_match = ""
        if ground_truth is not None:
            if result['pred_healed'] == ground_truth:
                healed_match = '<div class="metric-delta-up">CORRECT</div>'
            else:
                healed_match = '<div class="metric-delta-down">INCORRECT</div>'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{CLASS_NAMES[result['pred_healed']]}</div>
            <div class="metric-label">Final Diagnosis</div>
            {healed_match}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Probability Comparison Chart ──
    st.markdown("### Probability Breakdown")
    chart_df = pd.DataFrame({
        'Class': CLASS_NAMES * 3,
        'Probability': list(result['probs_clean']) + list(result['probs_noisy']) + list(result['probs_healed']),
        'Stage': ['Clean'] * 2 + ['Noisy'] * 2 + ['Healed'] * 2,
    })

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        noisy_df = pd.DataFrame({
            'NORMAL': [result['probs_noisy'][0]],
            'PNEUMONIA': [result['probs_noisy'][1]],
        }, index=['Corrupted'])
        st.bar_chart(noisy_df, color=["#22c55e", "#ef4444"], height=250)

    with chart_c2:
        healed_df = pd.DataFrame({
            'NORMAL': [result['probs_healed'][0]],
            'PNEUMONIA': [result['probs_healed'][1]],
        }, index=['Healed'])
        st.bar_chart(healed_df, color=["#22c55e", "#ef4444"], height=250)


# ─────────────────────────────────────────────────────────────
# FULL TEST SET EVAL
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    if st.button("Evaluate Full Test Set", use_container_width=True):
        st.session_state['run_eval'] = True

if st.session_state.get('run_eval'):
    st.session_state['run_eval'] = False
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Full Test Set Evaluation")

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct_clean, correct_noisy, correct_healed = 0, 0, 0
    total = 0
    per_class = {0: {'clean': 0, 'noisy': 0, 'healed': 0, 'total': 0},
                 1: {'clean': 0, 'noisy': 0, 'healed': 0, 'total': 0}}

    progress = st.progress(0, text="Evaluating...")
    n_batches = len(loader)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            noisy = add_noise(images, noise_factor)
            healed_imgs, _, _ = healer(noisy)

            preds_clean = expert(images).argmax(dim=1)
            preds_noisy = expert(noisy).argmax(dim=1)
            preds_healed = expert(healed_imgs).argmax(dim=1)

            for j in range(labels.size(0)):
                gt = labels[j].item()
                per_class[gt]['total'] += 1
                if preds_clean[j].item() == gt:
                    correct_clean += 1
                    per_class[gt]['clean'] += 1
                if preds_noisy[j].item() == gt:
                    correct_noisy += 1
                    per_class[gt]['noisy'] += 1
                if preds_healed[j].item() == gt:
                    correct_healed += 1
                    per_class[gt]['healed'] += 1
                total += 1

            progress.progress((i + 1) / n_batches, text=f"Batch {i+1}/{n_batches}")

    progress.empty()

    acc_clean = 100 * correct_clean / total
    acc_noisy = 100 * correct_noisy / total
    acc_healed = 100 * correct_healed / total

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Clean Accuracy", f"{acc_clean:.2f}%")
    with e2:
        st.metric("Noisy Accuracy", f"{acc_noisy:.2f}%", delta=f"{acc_noisy - acc_clean:.2f}%")
    with e3:
        st.metric("Healed Accuracy", f"{acc_healed:.2f}%", delta=f"+{acc_healed - acc_noisy:.2f}% recovery")
    with e4:
        st.metric("Total Samples", str(total))

    st.markdown("#### Per-Class Breakdown")
    breakdown_data = []
    for cls_idx in [0, 1]:
        cls_total = per_class[cls_idx]['total']
        if cls_total > 0:
            breakdown_data.append({
                'Class': CLASS_NAMES[cls_idx],
                'Samples': cls_total,
                'Clean': f"{100 * per_class[cls_idx]['clean'] / cls_total:.1f}%",
                'Noisy': f"{100 * per_class[cls_idx]['noisy'] / cls_total:.1f}%",
                'Healed': f"{100 * per_class[cls_idx]['healed'] / cls_total:.1f}%",
            })
    st.table(pd.DataFrame(breakdown_data))
