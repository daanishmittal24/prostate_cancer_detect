#!/bin/bash

# Exit on error
set -e

# Configuration
NUM_GPUS=4  # Set to the number of GPUs you want to use (0 to use all available)
PORT=29500  # Base port (will be incremented for each process)
DATA_DIR="/path/to/your/data"  # Update this path
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# If NUM_GPUS is 0 or greater than available, use all GPUs
if [ "$NUM_GPUS" -eq 0 ] || [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "Using $NUM_GPUS GPUs for training"

# Training parameters
BATCH_SIZE=16  # Per GPU batch size
EPOCHS=10
LEARNING_RATE=5e-5
MODEL_NAME="google/vit-base-patch16-224-in21k"

# Calculate total batch size (batch size per GPU * number of GPUs)
TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

echo "Total batch size: $TOTAL_BATCH_SIZE (${BATCH_SIZE} per GPU * ${NUM_GPUS} GPUs)"

# Log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Function to run training
run_training() {
    # The actual training command
    CMD="python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        train.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model-save-path "$OUTPUT_DIR/best_model.pth" \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --distributed \
        --dist-backend nccl \
        --dist-url 'tcp://127.0.0.1:$PORT'"
    
    echo "Starting training with command:"
    echo "$CMD"
    echo "Logging to: $LOG_FILE"
    
    # Run the command with nohup
    nohup $CMD > "$LOG_FILE" 2>&1 &
    
    # Get the process ID
    TRAIN_PID=$!
    
    echo "Training started with PID: $TRAIN_PID"
    echo "To view the logs, run: tail -f \"$LOG_FILE\""
    echo "To kill the process, run: kill $TRAIN_PID"
}

# Start the training
run_training

echo "Training is running in the background."
echo "Check the log file for progress: $LOG_FILE"
