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

# Step 2: Create folders
echo "📁 Step 2/4: Creating required folders..."
mkdir -p data/raw data/processed data/samples models outputs/plots outputs/results

# Step 3: Download ImageNet-100
echo "⬇️  Step 3/4: Downloading ImageNet-100 from Kaggle..."
echo "⚠️  Make sure your kaggle.json API key is in ~/.kaggle/"
kaggle datasets download -d ambityga/imagenet100 -p ./data/raw/
cd data/raw && unzip imagenet100.zip && cd ../..
echo "✅ Dataset downloaded and extracted"

# Step 4: Verify CUDA
echo "🔍 Step 4/4: Verifying CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup complete! Now run:"
echo "   python src/train_vae.py"
echo "   python src/train_classifier.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
