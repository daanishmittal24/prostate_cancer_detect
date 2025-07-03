#!/bin/bash

# System Diagnostic Script for Training
# This script checks if your system is ready for distributed training

echo "=== System Diagnostic for PyTorch Distributed Training ==="
echo "Date: $(date)"
echo

# Check Python and PyTorch installation
echo "1. Python and PyTorch Environment:"
echo "Python version: $(python --version 2>&1)"
echo "Python path: $(which python)"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  Not in a virtual environment"
fi

# Check PyTorch installation
python -c "
import sys
try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ PyTorch CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ Number of GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # Check distributed support
    try:
        import torch.distributed
        print('✅ torch.distributed is available')
        
        # Check if torchrun is available
        import subprocess
        try:
            subprocess.run(['torchrun', '--help'], capture_output=True, check=True)
            print('✅ torchrun is available')
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('❌ torchrun is not available')
            
    except ImportError:
        print('❌ torch.distributed is not available')
        
except ImportError as e:
    print(f'❌ PyTorch not found: {e}')
    sys.exit(1)
" 2>/dev/null
echo

# Check NVIDIA drivers and CUDA
echo "2. NVIDIA and CUDA Status:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
    echo
    echo "CUDA runtime info:"
    nvcc --version 2>/dev/null || echo "❌ nvcc not found"
else
    echo "❌ nvidia-smi not found - NVIDIA drivers may not be installed"
fi
echo

# Check other dependencies
echo "3. Other Dependencies:"
python -c "
packages = ['transformers', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'tqdm', 'cv2', 'PIL']
for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
            print(f'✅ opencv-python: {cv2.__version__}')
        elif pkg == 'PIL':
            import PIL
            print(f'✅ Pillow: {PIL.__version__}')
        else:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: not installed')
" 2>/dev/null
echo

# Check data directory
echo "4. Data Directory Check:"
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory exists: $DATA_DIR"
    echo "Contents:"
    ls -la "$DATA_DIR" | head -10
    
    # Check specific files
    if [ -f "$DATA_DIR/train.csv" ]; then
        echo "✅ train.csv found"
        echo "   Rows: $(wc -l < "$DATA_DIR/train.csv")"
    else
        echo "❌ train.csv not found"
    fi
    
    if [ -d "$DATA_DIR/train_images" ]; then
        echo "✅ train_images directory found"
        echo "   Images: $(ls "$DATA_DIR/train_images" | wc -l)"
    else
        echo "❌ train_images directory not found"
    fi
else
    echo "❌ Data directory not found: $DATA_DIR"
fi
echo

# Check disk space
echo "5. Disk Space:"
df -h . | head -2
echo

# Check memory
echo "6. System Memory:"
free -h 2>/dev/null || echo "free command not available"
echo

# Recommendations
echo "=== Recommendations ==="

# Check if we can run a simple distributed test
echo "7. Testing Distributed Setup:"
python -c "
import torch
import os

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print('✅ Multiple GPUs detected, distributed training possible')
    
    # Test NCCL backend
    try:
        if torch.distributed.is_nccl_available():
            print('✅ NCCL backend available')
        else:
            print('⚠️  NCCL backend not available, consider using gloo backend')
    except:
        print('⚠️  Could not check NCCL availability')
        
elif torch.cuda.is_available():
    print('⚠️  Only one GPU detected, distributed training not needed')
    print('   Consider running without --distributed flag')
else:
    print('❌ No CUDA GPUs detected, will run on CPU')
" 2>/dev/null

echo
echo "=== Quick Fixes ==="
echo "If you see errors above, try these fixes:"
echo
echo "1. If PyTorch is missing or outdated:"
echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo
echo "2. If torchrun is missing (older PyTorch):"
echo "   pip install --upgrade torch"
echo
echo "3. If you have fewer than 4 GPUs, update NUM_GPUS in run_on_server.sh:"
echo "   NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo
echo "4. If NCCL is not available, use gloo backend:"
echo "   Change --dist-backend nccl to --dist-backend gloo in run_on_server.sh"
echo
echo "5. For single GPU, run without distributed:"
echo "   python train.py --data-dir /path/to/data --batch-size 16 --epochs 10"
echo
