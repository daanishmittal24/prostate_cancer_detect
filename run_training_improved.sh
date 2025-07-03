#!/bin/bash

# Updated Run Script with Better Error Handling
# Exit on error
set -e

echo "=== Prostate Cancer Detection Training - Updated ==="
echo "Date: $(date)"
echo

# Configuration
NUM_GPUS=0  # Set to 0 to auto-detect, or specify number
PORT=29500  # Base port (will be incremented for each process)
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"  # Update this path
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"

# Training parameters
BATCH_SIZE=16  # Per GPU batch size
EPOCHS=10
LEARNING_RATE=5e-5
MODEL_NAME="google/vit-base-patch16-224-in21k"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script to point to your data"
    exit 1
fi

# Check PyTorch and CUDA
echo "1. Checking PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA not available, will use CPU')
"

# Get number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Available GPUs: $AVAILABLE_GPUS"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    AVAILABLE_GPUS=0
    echo "⚠️  nvidia-smi not found, assuming no GPUs available"
fi

# Set number of GPUs to use
if [ "$NUM_GPUS" -eq 0 ] || [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo
echo "2. Training Configuration:"
echo "   Using $NUM_GPUS GPUs for training"
echo "   Batch size per GPU: $BATCH_SIZE"
echo "   Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "   Epochs: $EPOCHS"
echo "   Learning rate: $LEARNING_RATE"
echo "   Data directory: $DATA_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo

# Log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Function to run training
run_training() {
    if [ "$NUM_GPUS" -le 1 ]; then
        echo "3. Running single GPU/CPU training..."
        CMD="python train.py \
            --data-dir \"$DATA_DIR\" \
            --output-dir \"$OUTPUT_DIR\" \
            --model-save-path \"$OUTPUT_DIR/best_model.pth\" \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --lr $LEARNING_RATE"
    else
        echo "3. Running distributed training with $NUM_GPUS GPUs..."
        
        # Check if torchrun is available
        if command -v torchrun &> /dev/null; then
            echo "Using torchrun (recommended)..."
            CMD="torchrun \
                --standalone \
                --nproc_per_node=$NUM_GPUS \
                --nnodes=1 \
                train.py \
                --data-dir \"$DATA_DIR\" \
                --output-dir \"$OUTPUT_DIR\" \
                --model-save-path \"$OUTPUT_DIR/best_model.pth\" \
                --batch-size $BATCH_SIZE \
                --epochs $EPOCHS \
                --lr $LEARNING_RATE \
                --distributed \
                --dist-backend nccl"
        else
            echo "torchrun not available, using torch.distributed.launch..."
            CMD="python -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPUS \
                --master_port=$PORT \
                --use_env \
                train.py \
                --data-dir \"$DATA_DIR\" \
                --output-dir \"$OUTPUT_DIR\" \
                --model-save-path \"$OUTPUT_DIR/best_model.pth\" \
                --batch-size $BATCH_SIZE \
                --epochs $EPOCHS \
                --lr $LEARNING_RATE \
                --distributed \
                --dist-backend nccl"
        fi
    fi
    
    echo "Starting training with command:"
    echo "$CMD"
    echo "Logging to: $LOG_FILE"
    echo
    
    # Run the command with nohup
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    
    # Get the process ID
    TRAIN_PID=$!
    
    echo "✅ Training started with PID: $TRAIN_PID"
    echo
    echo "📋 To monitor training:"
    echo "   tail -f \"$LOG_FILE\""
    echo
    echo "🔄 To check GPU usage:"
    echo "   watch -n 1 nvidia-smi"
    echo
    echo "⏹️  To stop training:"
    echo "   kill $TRAIN_PID"
    echo "   # Or use the stop script: ./stop_training.sh"
    echo
    
    # Save PID for easy reference
    echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
    echo "PID saved to: $OUTPUT_DIR/train.pid"
}

# Check if training is already running
EXISTING_PIDS=$(ps aux | grep -E "(torch.distributed.launch|torchrun.*train.py|python.*train.py)" | grep -v grep | awk '{print $2}')
if [ -n "$EXISTING_PIDS" ]; then
    echo "⚠️  Warning: Training processes already running with PIDs: $EXISTING_PIDS"
    echo "Do you want to continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Start the training
run_training

echo
echo "🚀 Training is running in the background."
echo "📄 Check the log file for progress: $LOG_FILE"
echo
echo "💡 Quick tip: You can run './monitor_training.sh' to see a training dashboard"
