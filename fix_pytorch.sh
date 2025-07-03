#!/bin/bash

# Fix PyTorch Installation Script
# This script helps you install PyTorch with distributed support

echo "=== PyTorch Installation Fix ==="
echo "Date: $(date)"
echo

# Check if we should proceed
echo "This script will reinstall PyTorch with full distributed support."
echo "Do you want to continue? (y/N)"
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo "Starting PyTorch installation..."
echo

# Determine Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "Using Python: $PYTHON_CMD"
echo "Using pip: $PIP_CMD"
echo

# Step 1: Uninstall existing PyTorch
echo "1. Uninstalling existing PyTorch packages..."
$PIP_CMD uninstall -y torch torchvision torchaudio || echo "Some packages were not installed"
echo

# Step 2: Install PyTorch with CUDA support
echo "2. Installing PyTorch with CUDA 11.8 support..."
$PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if [ $? -eq 0 ]; then
    echo "   ✅ PyTorch installation completed"
else
    echo "   ❌ PyTorch installation failed"
    echo "   Trying alternative installation..."
    $PIP_CMD install torch torchvision torchaudio
fi
echo

# Step 3: Install other requirements
echo "3. Installing other required packages..."
$PIP_CMD install -r requirements.txt
echo

# Step 4: Test installation
echo "4. Testing PyTorch installation..."
echo "Testing basic PyTorch import..."
if $PYTHON_CMD -c "import torch; print('✅ PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "Basic PyTorch import successful"
else
    echo "❌ Basic PyTorch import failed"
    exit 1
fi

echo "Testing CUDA availability..."
$PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

echo "Testing distributed module..."
if $PYTHON_CMD -c "import torch.distributed; print('✅ torch.distributed module available')" 2>/dev/null; then
    echo "Distributed module test successful"
else
    echo "❌ Distributed module test failed"
    echo "This may indicate an incomplete installation"
fi

echo "Testing transformers..."
if $PYTHON_CMD -c "import transformers; print('✅ Transformers version:', transformers.__version__)" 2>/dev/null; then
    echo "Transformers test successful"
else
    echo "❌ Transformers test failed"
fi
echo

# Step 5: Final verification
echo "5. Final verification..."
echo "Running complete setup check..."
./check_setup.sh

echo
echo "=== INSTALLATION COMPLETE ==="
echo
echo "Next steps:"
echo "1. If all tests passed, you can now run: ./run_on_server.sh"
echo "2. If there are still issues, check the error messages above"
echo "3. Make sure you're using the correct Python environment"
echo
echo "Done."
