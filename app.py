import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Self-Healing AI", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>Self-Healing Neural Network</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2em; color: #aaaaaa;'>Watch a Variational Autoencoder (VAE) recover heavily corrupted digits before feeding them to a Convolutional Neural Network (CNN).</p>", unsafe_allow_html=True)
st.divider()

# --- 2. MODEL BLUEPRINTS ---
# (We have to include the blueprints so PyTorch knows how to load the saved brains)
class DenoisingVAE(nn.Module):
    def __init__(self):
        super(DenoisingVAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, 20)
        self.fc_logvar = nn.Linear(64 * 7 * 7, 20)
        self.fc_decode = nn.Linear(20, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x = F.relu(self.fc_decode(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.dec_conv1(x))
        return torch.sigmoid(self.dec_conv2(x)), mu, logvar

class ExpertCNN(nn.Module):
    def __init__(self):
        super(ExpertCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- 3. LOAD MODELS (Cached so it only happens once) ---
@st.cache_resource
def load_models():
    healer = DenoisingVAE()
    expert = ExpertCNN()
    # Assuming app.py is in the main folder, the path is just './models/...'
    healer.load_state_dict(torch.load('./models/healer_vae.pth', map_location=torch.device('cpu'), weights_only=True))
    expert.load_state_dict(torch.load('./models/expert_cnn.pth', map_location=torch.device('cpu'), weights_only=True))
    healer.eval()
    expert.eval()
    return healer, expert

healer, expert = load_models()

# --- 4. NOISE FUNCTION ---
def add_noise(image_tensor, factor):
    noise = torch.randn_like(image_tensor) * factor
    return torch.clamp(image_tensor + noise, 0., 1.)

# --- 5. ENHANCED USER INTERFACE ---
# Sidebar Controls
with st.sidebar:
    st.header("Pipeline Controls")
    noise_level = st.slider("Static Noise Intensity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    st.info("Higher noise makes it harder for the CNN to guess. Let's see if the VAE Healer can filter it out!")
    st.divider()
    st.markdown("### Architecture")
    st.markdown("- **Healer:** PyTorch VAE\n- **Expert:** PyTorch CNN")

col_draw, col_results = st.columns([1, 2.2])

with col_draw:
    st.markdown("### 1. Draw a Digit (0-9)")
    st.markdown("Draw as thick and clear as possible.")
    # Create a canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    process_button = st.button("Heal & Predict", type="primary", use_container_width=True)

with col_results:
    st.markdown("### 2. Pipeline Analysis")
    
    if process_button and canvas_result.image_data is not None:
        # Process the drawing
        img_array = canvas_result.image_data
        gray_image = np.mean(img_array[:, :, :3], axis=2) 
        
        pil_img = Image.fromarray(gray_image.astype('uint8'))
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        small_img_array = np.array(pil_img) / 255.0
        clean_tensor = torch.tensor(small_img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        if clean_tensor.max() > 0:
            with st.spinner('Neural Networks are processing the image...'):
                with torch.no_grad():
                    # Step B: Inject Noise
                    noisy_tensor = add_noise(clean_tensor, noise_level)
                    
                    # Step C: Heal the Image
                    healed_tensor, _, _ = healer(noisy_tensor)
                    
                    # Step D: Get Predictions and Probabilities
                    logits_noisy = expert(noisy_tensor)
                    probs_noisy = F.softmax(logits_noisy, dim=1).squeeze().numpy()
                    pred_noisy = np.argmax(probs_noisy)
                    
                    logits_healed = expert(healed_tensor)
                    probs_healed = F.softmax(logits_healed, dim=1).squeeze().numpy()
                    pred_healed = np.argmax(probs_healed)
            
            # --- VISUALIZE IMAGES ---
            st.markdown("#### Vision Stages")
            img_col1, img_col2, img_col3 = st.columns(3)
            
            with img_col1:
                st.image(clean_tensor.squeeze().numpy(), caption="1. Original Input", use_container_width=True)
            with img_col2:
                st.image(noisy_tensor.squeeze().numpy(), caption=f"2. Corrupted (Noise: {noise_level:.2f})", use_container_width=True)
            with img_col3:
                st.image(healed_tensor.squeeze().numpy(), caption="3. VAE Healed Output", use_container_width=True)
            
            st.divider()
            
            # --- VISUALIZE METRICS & PROBABILITIES ---
            st.markdown("#### Expert CNN Thought Process")
            stat_col1, stat_col2 = st.columns(2)
            
            # Prepare chart data
            classes = [str(i) for i in range(10)]
            df_noisy = pd.DataFrame({"Probability": probs_noisy}, index=classes)
            df_healed = pd.DataFrame({"Probability": probs_healed}, index=classes)
            
            with stat_col1:
                if pred_noisy != pred_healed:
                    st.metric(label="Prediction on Noisy Data", value=pred_noisy, delta="Failed", delta_color="inverse")
                else:
                    st.metric(label="Prediction on Noisy Data", value=pred_noisy, delta="Struggled but Correct", delta_color="off")
                st.bar_chart(df_noisy, y="Probability", color="#ff4b4b", height=200)

            with stat_col2:
                st.metric(label="Prediction on Healed Data", value=pred_healed, delta="Confident & Correct!" if probs_healed[pred_healed] > 0.8 else "Healed")
                st.bar_chart(df_healed, y="Probability", color="#00cc66", height=200)

        else:
            st.warning("The canvas is empty. Please draw a digit first!")
    elif not process_button:
        st.info("Draw a digit on the canvas and click 'Heal & Predict' to see the magic!")