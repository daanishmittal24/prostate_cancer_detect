#!/bin/bash

# Setup Diagnostic Script
# This script checks your Python environment and PyTorch installation

echo "=== Python Environment & PyTorch Setup Check ==="
echo "Date: $(date)"
echo

# 1. Check Python version and location
echo "1. Python Information:"
echo "   Python version: $(python3 --version 2>/dev/null || echo 'python3 not found')"
echo "   Python location: $(which python3 2>/dev/null || echo 'python3 not found')"
echo "   Python (python): $(python --version 2>/dev/null || echo 'python not found')"
echo "   Python location: $(which python 2>/dev/null || echo 'python not found')"
echo

# 2. Check if we're in a virtual environment
echo "2. Virtual Environment Check:"
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "   ✅ Virtual environment active: $VIRTUAL_ENV"
elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "   ✅ Conda environment active: $CONDA_DEFAULT_ENV"
else
    echo "   ⚠️  No virtual environment detected"
    echo "   Consider activating a virtual environment or conda environment"
fi
echo

# 3. Check PyTorch installation
echo "3. PyTorch Installation Check:"
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if torch is installed
if $PYTHON_CMD -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "   ✅ PyTorch is installed"
    
    # Check CUDA availability
    if $PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        echo "   ✅ CUDA check completed"
        $PYTHON_CMD -c "import torch; print('   CUDA devices:', torch.cuda.device_count())" 2>/dev/null
    fi
    
    # Check distributed module
    if $PYTHON_CMD -c "import torch.distributed" 2>/dev/null; then
        echo "   ✅ torch.distributed module is available"
    else
        echo "   ❌ torch.distributed module is NOT available"
        echo "   This is the main issue causing your training to fail!"
    fi
    
else
    echo "   ❌ PyTorch is NOT installed"
fi
echo

# 4. Check other required packages
echo "4. Required Packages Check:"
for package in transformers pandas numpy scikit-learn matplotlib tqdm opencv-python pillow scipy; do
    if $PYTHON_CMD -c "import ${package//-/_}" 2>/dev/null; then
        echo "   ✅ $package"
    else
        echo "   ❌ $package"
    fi
done
echo

# 5. Check GPU availability
echo "5. GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi available"
    echo "   GPU details:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    echo "   ❌ nvidia-smi not found - GPU drivers may not be installed"
fi
echo

# 6. Check if training processes are running
echo "6. Current Training Status:"
TRAIN_PIDS=$(ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep | awk '{print $2}')
if [ -z "$TRAIN_PIDS" ]; then
    echo "   ✅ No training processes currently running"
else
    echo "   ⚠️  Training processes found: $TRAIN_PIDS"
    echo "   Use stop_training.sh to stop them before restarting"
fi
echo

# 7. Provide solutions
echo "=== SOLUTIONS ==="
echo

if ! $PYTHON_CMD -c "import torch.distributed" 2>/dev/null; then
    echo "🔧 MAIN ISSUE: torch.distributed not available"
    echo
    echo "Solution 1 - Reinstall PyTorch with full features:"
    echo "   pip uninstall torch torchvision"
    echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo
    echo "Solution 2 - If using conda:"
    echo "   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo
    echo "Solution 3 - Install from requirements.txt:"
    echo "   pip install -r requirements.txt"
    echo
fi

echo "🔧 RECOMMENDED SETUP STEPS:"
echo "1. Stop any running training: ./stop_training.sh"
echo "2. Create/activate virtual environment:"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # Linux/Mac"
echo "   # or venv\\Scripts\\activate  # Windows"
echo "3. Install requirements: pip install -r requirements.txt"
echo "4. Run this check again: ./check_setup.sh"
echo "5. Start training: ./run_on_server.sh"
echo

echo "Done."
