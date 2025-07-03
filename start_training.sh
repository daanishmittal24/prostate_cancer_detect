#!/bin/bash

# Quick Start Training Script for SSH Server
echo "=== Quick Start Training on Server ==="
echo "Date: $(date)"
echo "Server: $(hostname)"
echo

# Configuration
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
EPOCHS=10
LEARNING_RATE=5e-5

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Model: Local ViT (./local_vit_model)"
echo

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Available directories in /home/Saif/Pratham/ELC/:"
    ls -la /home/Saif/Pratham/ELC/ 2>/dev/null || echo "Cannot list directory"
    echo
    echo "Please update DATA_DIR in this script to the correct path"
    exit 1
fi

echo "✅ Data directory found"
echo "Data contents:"
ls -la "$DATA_DIR" | head -5

# Check if we have the local model
if [ ! -d "./local_vit_model" ]; then
    echo "❌ Local ViT model not found. Please run ./download_model.sh first"
    exit 1
fi

echo "✅ Local ViT model found"

# Log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo
echo "Starting single GPU training..."
echo "Logging to: $LOG_FILE"
echo

# Training command (single GPU, no distributed)
CMD="python train.py \
    --data-dir \"$DATA_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --model-save-path \"$OUTPUT_DIR/best_model.pth\" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE"

echo "Command: $CMD"
echo

# Start training in background
nohup $CMD > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✅ Training started successfully!"
echo "📊 Process ID: $TRAIN_PID"
echo "📄 Log file: $LOG_FILE"
echo

# Save PID for easy reference
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "=== How to Monitor Training ==="
echo "1. View real-time logs:"
echo "   tail -f \"$LOG_FILE\""
echo
echo "2. Check GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo
echo "3. Check if process is running:"
echo "   ps aux | grep $TRAIN_PID"
echo
echo "4. Stop training:"
echo "   kill $TRAIN_PID"
echo
echo "5. Check training progress:"
echo "   ls -la \"$OUTPUT_DIR\""
echo

# Show initial log output
sleep 3
echo "=== Initial Training Output ==="
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE"
else
    echo "Log file not created yet, wait a moment..."
fi

echo
echo "🚀 Training is now running in the background!"
echo "💡 To monitor: tail -f \"$LOG_FILE\""
