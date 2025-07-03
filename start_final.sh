#!/bin/bash

# Final Startup Script - Uses Fixed train.py
echo "=== Final Training Setup ==="
echo "Date: $(date)"
echo "Server: $(hostname)"
echo

# Server Configuration
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_final_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
EPOCHS=10

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo

# Check data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Available directories:"
    ls -la /home/Saif/Pratham/ELC/ 2>/dev/null || echo "Cannot list directory"
    exit 1
fi

echo "✅ Data directory found"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi
echo

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Starting training with fixed train.py..."
echo "Logging to: $LOG_FILE"
echo

# Training command - using the fixed train.py
CMD="python3 train.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 5e-5"

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

echo "=== Quick Monitoring ==="
echo "1. View logs: tail -f \"$LOG_FILE\""
echo "2. Check GPU: watch -n 1 nvidia-smi"
echo "3. Stop training: kill $TRAIN_PID"
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
echo "🚀 Training is running with fixed Python 3.6 compatible code!"
