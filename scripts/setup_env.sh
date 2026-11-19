#!/bin/bash
set -e

echo "Setting up CodeQA Qwen2.5-Coder reproduction environment..."

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (adjust index-url according to GPU/CUDA if needed)
echo "Installing PyTorch & dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || pip install torch torchvision torchaudio

# Install HuggingFace & ML libraries
pip install transformers datasets accelerate peft sacrebleu rouge-score nltk matplotlib pandas pyyaml pyarrow

echo "Environment setup complete!"
