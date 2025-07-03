#!/bin/bash

# Minimal Training Startup Script for SSH Server
echo "=== Minimal Training Setup ==="
echo "Date: $(date)"
echo "Server: $(hostname)"
echo

# Configuration - Update these paths for your server
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=5e-5

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script"
    exit 1
fi

echo "✅ Data directory found"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found, using CPU"
fi
echo

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Starting training..."
echo "Logging to: $LOG_FILE"
echo

# Training command
CMD="python3 train_minimal.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE"

echo "Command: $CMD"
echo

# Start training
nohup $CMD > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✅ Training started!"
echo "📊 Process ID: $TRAIN_PID"
echo "📄 Log file: $LOG_FILE"
echo

# Save PID
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "=== Monitoring Commands ==="
echo "1. View logs: tail -f \"$LOG_FILE\""
echo "2. Check GPU: watch -n 1 nvidia-smi"
echo "3. Stop training: kill $TRAIN_PID"
echo "4. Check outputs: ls -la \"$OUTPUT_DIR\""
echo

# Show initial output
sleep 3
echo "=== Initial Output ==="
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE"
else
    echo "Log file not created yet..."
fi

echo
echo "🚀 Training is running!"
