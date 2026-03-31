#!/bin/bash
# ═══════════════════════════════════════════════
# setup_training.sh
# Run this script on Friend's Laptop (LOQ 3050)
# Usage: bash setup_training.sh
# ═══════════════════════════════════════════════

echo "🚀 Setting up Self-Healing Neural Network Training Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1: Install dependencies
echo "📦 Step 1/4: Installing Python dependencies..."
pip install -r requirements-train.txt
echo "📦 Installing CUDA-enabled PyTorch (cu121) for RTX 3050..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Create folders
echo "📁 Step 2/4: Creating required folders..."
mkdir -p data/raw data/processed data/samples models outputs/plots outputs/results

# Step 3: Trigger CIFAR-100 auto-download
echo "⬇️  Step 3/4: Downloading CIFAR-100 via torchvision..."
python -c "from torchvision import datasets; datasets.CIFAR100(root='data/raw', train=True, download=True); datasets.CIFAR100(root='data/raw', train=False, download=True); print('CIFAR-100 downloaded')"
echo "✅ CIFAR-100 downloaded"

# Step 4: Verify CUDA
echo "🔍 Step 4/4: Verifying CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete! Now run:"
echo "   python -m src.train_vae"
echo "   python -m src.train_classifier"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
