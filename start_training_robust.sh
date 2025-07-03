#!/bin/bash

# Robust Training Script with Error Handling
echo "=== Robust Training Script ==="
echo "This version handles corrupted images gracefully"
echo

# Configuration (same as original)
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./outputs_robust_$(date +%Y%m%d_%H%M%S)"
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

# Check for cleaned dataset first
CLEAN_CSV="$DATA_DIR/train_clean.csv"
if [ -f "$CLEAN_CSV" ]; then
    echo "✅ Found cleaned dataset: train_clean.csv"
    USE_CLEAN="--train-csv train_clean.csv"
else
    echo "⚠️  No cleaned dataset found, will use original with error handling"
    USE_CLEAN=""
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "✅ Data directory found"

# Check if we have the local model
if [ ! -d "./local_vit_model" ]; then
    echo "❌ Local ViT model not found. Please run ./download_model.sh first"
    exit 1
fi

echo "✅ Local ViT model found"

# Log file
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"

echo
echo "Starting robust training with error handling..."
echo "Logging to: $LOG_FILE"
echo

# Enhanced training command with error handling
CMD="python train_robust.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --model-save-path $OUTPUT_DIR/best_model.pth \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    $USE_CLEAN"

echo "Command: $CMD"
echo

# Check if we have the robust training script
if [ ! -f "train_robust.py" ]; then
    echo "Creating robust training script..."
    cp train.py train_robust.py
    echo "✅ Created train_robust.py"
fi

# Start training in background
nohup $CMD > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✅ Robust training started successfully!"
echo "📊 Process ID: $TRAIN_PID"
echo "📄 Log file: $LOG_FILE"
echo

# Save PID for easy reference
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "=== Monitoring ==="
echo "1. View live logs: tail -f \"$LOG_FILE\""
echo "2. Check GPU: watch -n 1 nvidia-smi"
echo "3. Stop training: kill $TRAIN_PID"
echo

# Show initial log output
sleep 3
echo "=== Initial Training Output ==="
if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo "Log file not created yet, wait a moment..."
fi

echo
echo "🚀 Robust training is running!"
