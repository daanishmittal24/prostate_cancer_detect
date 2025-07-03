#!/bin/bash

# Environment Setup Script
echo "=== Environment Setup for Simple Training ==="
echo

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
    echo "Recommended: create and activate a virtual environment first"
    echo "Example: python3 -m venv cancer_env && source cancer_env/bin/activate"
    echo
fi

# Check Python version
echo "Python version:"
python3 --version

# Install requirements
echo
echo "Installing required packages..."
pip3 install --upgrade pip
pip3 install -r requirements_simple.txt

# Verify installations
echo
echo "Verifying installations..."

echo "Checking PyTorch..."
python3 -c "
import torch
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA version:', torch.version.cuda)
    print('✅ GPU count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print('  GPU {}: {}'.format(i, torch.cuda.get_device_name(i)))
"

echo "Checking other packages..."
python3 -c "
try:
    import pandas as pd
    print('✅ Pandas version:', pd.__version__)
except ImportError:
    print('❌ Pandas not installed')

try:
    import PIL
    print('✅ Pillow version:', PIL.__version__)
except ImportError:
    print('❌ Pillow not installed')

try:
    import transformers
    print('✅ Transformers version:', transformers.__version__)
except ImportError:
    print('❌ Transformers not installed')

try:
    import numpy as np
    print('✅ NumPy version:', np.__version__)
except ImportError:
    print('❌ NumPy not installed')

try:
    import tqdm
    print('✅ tqdm available')
except ImportError:
    print('❌ tqdm not installed')
"

echo
echo "Environment setup complete!"
echo "Run ./start_simple.sh to begin training"
