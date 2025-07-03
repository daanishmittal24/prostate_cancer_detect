#!/bin/bash

# Simple Monitor Script
echo "=== Training Monitor ==="
echo "Server: $(hostname)"
echo "Date: $(date)"
echo

# Find latest output directory
LATEST_OUTPUT=$(ls -td outputs_* 2>/dev/null | head -1)

if [ -z "$LATEST_OUTPUT" ]; then
    echo "❌ No training outputs found"
    exit 1
fi

echo "📁 Latest training: $LATEST_OUTPUT"
echo

# Check if training is running
if [ -f "$LATEST_OUTPUT/train.pid" ]; then
    PID=$(cat "$LATEST_OUTPUT/train.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Training is running (PID: $PID)"
    else
        echo "❌ Training process not found (PID: $PID)"
    fi
else
    echo "⚠️  No PID file found"
fi

# Show GPU status
echo
echo "🔥 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

# Show log file info
LOG_FILE="$LATEST_OUTPUT/training.log"
if [ -f "$LOG_FILE" ]; then
    echo
    echo "📄 Log file: $LOG_FILE"
    echo "📏 Log size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "⏰ Last modified: $(stat -c %y "$LOG_FILE")"
    
    echo
    echo "🔍 Recent log output (last 15 lines):"
    echo "----------------------------------------"
    tail -15 "$LOG_FILE"
    echo "----------------------------------------"
else
    echo
    echo "❌ Log file not found: $LOG_FILE"
fi

# Show metrics if available
METRICS_FILE="$LATEST_OUTPUT/metrics.json"
if [ -f "$METRICS_FILE" ]; then
    echo
    echo "📊 Training Progress:"
    echo "Epochs completed: $(jq '.train | length' "$METRICS_FILE" 2>/dev/null || echo "0")"
    
    # Show latest metrics
    LATEST_VAL_LOSS=$(jq -r '.val[-1].loss' "$METRICS_FILE" 2>/dev/null)
    LATEST_VAL_ACC=$(jq -r '.val[-1].accuracy' "$METRICS_FILE" 2>/dev/null)
    if [ "$LATEST_VAL_LOSS" != "null" ]; then
        echo "Latest val loss: $LATEST_VAL_LOSS"
        echo "Latest val accuracy: $LATEST_VAL_ACC"
    fi
fi

echo
echo "=== Quick Commands ==="
echo "📊 View live logs: tail -f \"$LOG_FILE\""
echo "🔄 Watch GPU: watch -n 1 nvidia-smi"
echo "⏹️  Stop training: kill \$(cat \"$LATEST_OUTPUT/train.pid\")"
echo "📁 Check outputs: ls -la \"$LATEST_OUTPUT\""
