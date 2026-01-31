#!/bin/bash
# Quick Installation Script for Lahaja Bengali Voice Cloning Pipeline
# Run this in your terminal to set up everything automatically
# Usage: bash install.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Lahaja Bengali Voice Cloning - Automatic Setup            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv lahaja_env
echo "   ✓ Virtual environment created: lahaja_env"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source lahaja_env/Scripts/activate
else
    source lahaja_env/bin/activate
fi
echo "   ✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "   ✓ pip upgraded"

# Install PyTorch
echo ""
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   🎮 GPU detected! Installing CUDA version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
    echo "   ✓ PyTorch with CUDA installed"
else
    echo "   💻 CPU mode. Installing CPU version..."
    pip install torch torchvision torchaudio > /dev/null 2>&1
    echo "   ✓ PyTorch with CPU installed"
fi

# Install other dependencies
echo ""
echo "📚 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "   ✓ All dependencies installed"
else
    echo "   ⚠️  requirements.txt not found. Installing manually..."
    pip install -q datasets transformers librosa soundfile scipy numpy pandas huggingface-hub
    echo "   ✓ Dependencies installed"
fi

# HuggingFace login
echo ""
echo "🔐 HuggingFace Hub Authentication"
echo "   To use Lahaja dataset and IndicF5 model, you need a HuggingFace token"
echo "   1. Go to: https://huggingface.co/settings/tokens"
echo "   2. Create a new token"
echo "   3. Copy the token"
echo ""
read -p "   Do you want to login to HuggingFace now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
    echo "   ✓ HuggingFace authentication complete"
else
    echo "   ⏭️  Skipping HuggingFace login (you can do it later with: huggingface-cli login)"
fi

# Create output directories
echo ""
echo "📁 Creating output directories..."
mkdir -p lahaja_bengali_cloned/audio
echo "   ✓ Output directories created"

# Final summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Installation Complete!                               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next Steps:"
echo ""
echo "   1. Activate environment (if not already active):"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "      lahaja_env\\Scripts\\activate"
else
    echo "      source lahaja_env/bin/activate"
fi
echo ""
echo "   2. Run the pipeline:"
echo "      python lahaja_bengali_voice_clone.py"
echo ""
echo "   3. Check output:"
echo "      ls -la lahaja_bengali_cloned/"
echo ""
echo "📚 Documentation:"
echo "   - Quick start: QUICKSTART.md"
echo "   - Full setup: setup_guide.md"
echo "   - Architecture: ARCHITECTURE_DIAGRAM.md"
echo "   - Advanced: advanced_examples.py"
echo ""
echo "🎯 Total output will be: 50 Bengali audio files @ 16kHz + metadata"
echo ""
echo "⏱️  Estimated runtime:"
echo "   - CPU mode: 20-40 minutes"
echo "   - GPU mode: 8-17 minutes"
echo ""
echo "💡 Tip: First run downloads IndicF5 model (~5-10 min) - subsequent runs are faster"
echo ""
echo "✨ Happy voice cloning! ✨"
echo ""
