#!/bin/bash

# Set the path to your Python script
TRAIN_SCRIPT="train.py"

# Set the path to your data directory
DATA_DIR="/path/to/your/data"

# Set the output directory
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Number of GPUs to use (set to 0 to use all available GPUs)
NUM_GPUS=0

# If NUM_GPUS is 0, use all available GPUs
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

echo "Using $NUM_GPUS GPUs for training"

# Set the port for distributed training (can be any available port)
PORT=12345

# Training parameters
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=5e-5
MODEL_SAVE_PATH="$OUTPUT_DIR/best_model.pth"

# Log file for nohup output
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Function to run training
run_training() {
    local cmd="python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$PORT \
        $TRAIN_SCRIPT \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --model-save-path "$MODEL_SAVE_PATH" \
        --batch-size $((BATCH_SIZE * NUM_GPUS)) \
        --epochs $EPOCHS \
        --lr $LEARNING_RATE \
        --distributed"
    
    echo "Starting training with command:"
    echo "$cmd"
    echo "Logging to: $LOG_FILE"
    
    # Run the command with nohup
    nohup $cmd > "$LOG_FILE" 2>&1 &
    
    # Get the process ID
    TRAIN_PID=$!
    
    echo "Training started with PID: $TRAIN_PID"
    echo "To view the logs, run: tail -f \"$LOG_FILE\""
    echo "To kill the process, run: kill $TRAIN_PID"
}

# Check if GPUs are available
if command -v nvidia-smi &> /dev/null; then
    echo "Found $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPU(s)"
    run_training
else
    echo "No GPUs found. Falling back to CPU training."
    NUM_GPUS=0
    run_training
fi
