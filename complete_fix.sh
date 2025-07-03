#!/bin/bash

# Complete Fix Script for Training Issues
echo "=== Complete Training Fix Script ==="
echo "This script addresses all the issues found in your training setup"
echo

# Step 1: Check system
echo "1. Checking system configuration..."
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "⚠️  No GPUs detected - will use CPU training"
elif [ "$AVAILABLE_GPUS" -eq 1 ]; then
    echo "✅ Single GPU detected - will use single GPU training"
else
    echo "✅ Multiple GPUs detected - distributed training possible"
fi

# Step 2: Update environment
echo
echo "2. Setting up environment..."

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Not in virtual environment"
    if [ -d "venv" ]; then
        echo "Activating existing virtual environment..."
        source venv/bin/activate
    else
        echo "Creating new virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    fi
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi

# Step 3: Install/upgrade dependencies
echo
echo "3. Checking and installing dependencies..."

# Upgrade transformers to fix download issues
pip install --upgrade transformers torch torchvision

# Install other requirements
pip install -r requirements.txt

# Step 4: Pre-download model
echo
echo "4. Pre-downloading model to avoid network issues during training..."
./download_model.sh

# Step 5: Update configuration files
echo
echo "5. Updating configuration for single GPU..."

# Update run script for single GPU
if [ "$AVAILABLE_GPUS" -le 1 ]; then
    echo "Configuring for single GPU training..."
    
    # Create a single GPU version of the run script
    cat > run_single_gpu.sh << 'EOF'
#!/bin/bash

# Single GPU Training Script
set -e

echo "=== Single GPU Training ==="
echo "Date: $(date)"

# Configuration
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=5e-5

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script"
    exit 1
fi

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR" 
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"

# Log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting single GPU training..."
echo "Logging to: $LOG_FILE"

# Run training command
nohup python train.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --model-save-path "$OUTPUT_DIR/best_model.pth" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training started with PID: $TRAIN_PID"
echo "📄 To monitor: tail -f \"$LOG_FILE\""
echo "⏹️  To stop: kill $TRAIN_PID"

# Save PID
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
echo "PID saved to: $OUTPUT_DIR/train.pid"
EOF

    chmod +x run_single_gpu.sh
    echo "✅ Created run_single_gpu.sh"
fi

# Step 6: Test setup
echo
echo "6. Testing setup..."
python -c "
import torch
from transformers import ViTModel
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

# Test model loading
import os
try:
    if os.path.exists('working_model.txt'):
        with open('working_model.txt', 'r') as f:
            model_name = f.read().strip()
    else:
        model_name = 'google/vit-base-patch16-224'
    
    print(f'Testing model: {model_name}')
    model = ViTModel.from_pretrained(model_name)
    print('✅ Model loading successful')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Tests failed - please check the errors above"
    exit 1
fi

# Step 7: Instructions
echo
echo "=== Next Steps ==="
echo
echo "1. Update the data directory path:"
if [ "$AVAILABLE_GPUS" -le 1 ]; then
    echo "   Edit DATA_DIR in run_single_gpu.sh"
    echo
    echo "2. Start training:"
    echo "   ./run_single_gpu.sh"
else
    echo "   Edit DATA_DIR in run_training_improved.sh"
    echo
    echo "2. Start training:"
    echo "   ./run_training_improved.sh"
fi

echo
echo "3. Monitor training:"
echo "   ./monitor_training.sh"
echo
echo "4. Stop training if needed:"
echo "   ./stop_training.sh"
echo
echo "✅ Setup complete!"
