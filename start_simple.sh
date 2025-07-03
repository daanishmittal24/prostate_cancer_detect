#!/bin/bash

# Simple Training Launcher
echo "=== Simple Training Script ==="
echo "Minimal setup for prostate cancer detection"
echo "Date: $(date)"
echo "Server: $(hostname)"
echo

# Configuration - EDIT THESE PATHS
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_simple_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=5e-5

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Available directories:"
    ls -la /home/Saif/Pratham/ELC/ 2>/dev/null || echo "Cannot list directory"
    echo
    echo "Please update DATA_DIR in this script"
    exit 1
fi

echo "✅ Data directory found"

# Check for train.csv
if [ ! -f "$DATA_DIR/train.csv" ]; then
    echo "❌ Error: train.csv not found in $DATA_DIR"
    echo "Contents of data directory:"
    ls -la "$DATA_DIR" | head -10
    exit 1
fi

echo "✅ train.csv found"

# Check if train_images directory exists
if [ ! -d "$DATA_DIR/train_images" ]; then
    echo "❌ Error: train_images directory not found in $DATA_DIR"
    echo "Contents of data directory:"
    ls -la "$DATA_DIR" | head -10
    exit 1
fi

echo "✅ train_images directory found"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check Python and packages
echo "Checking Python environment..."
python3 --version
echo "Checking PyTorch..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
fi

echo
echo "Starting simple training..."

# Training command
CMD="python3 train_simple.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE"

echo "Command: $CMD"
echo

# Create log file
LOG_FILE="$OUTPUT_DIR/training.log"

# Start training
echo "Training output will be logged to: $LOG_FILE"
$CMD 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo
    echo "✅ Training completed successfully!"
    echo "📁 Output directory: $OUTPUT_DIR"
    echo "📄 Log file: $LOG_FILE"
    echo "🏆 Best model: $OUTPUT_DIR/best_model.pth"
    echo "📊 Metrics: $OUTPUT_DIR/metrics.json"
else
    echo
    echo "❌ Training failed! Check the log file: $LOG_FILE"
    echo "Last 20 lines of log:"
    tail -20 "$LOG_FILE"
fi
