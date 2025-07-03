#!/bin/bash

# Monitor Training Script
# This script helps you monitor the training process

echo "=== Training Monitoring Dashboard ==="
echo "Date: $(date)"
echo

# Check if training process is running
echo "1. Checking training processes..."
TRAIN_PROCESSES=$(ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep)
if [ -n "$TRAIN_PROCESSES" ]; then
    echo "✅ Training processes found:"
    echo "$TRAIN_PROCESSES"
else
    echo "❌ No training processes found"
fi
echo

# Check GPU status
echo "2. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "❌ nvidia-smi not available"
fi
echo

# Check latest output directory
echo "3. Latest output directory:"
LATEST_OUTPUT=$(ls -td outputs_* 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "📁 $LATEST_OUTPUT"
    
    # Check log file
    LOG_FILE=$(ls -t "$LATEST_OUTPUT"/training_*.log 2>/dev/null | head -1)
    if [ -f "$LOG_FILE" ]; then
        echo "📄 Log file: $LOG_FILE"
        echo "📊 Log file size: $(du -h "$LOG_FILE" | cut -f1)"
        echo "⏰ Last modified: $(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)"
        
        # Show last few lines of log
        echo
        echo "4. Last 10 lines of training log:"
        echo "----------------------------------------"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "Could not read log file"
        echo "----------------------------------------"
    else
        echo "❌ No log file found in $LATEST_OUTPUT"
    fi
    
    # Check checkpoints
    if [ -d "$LATEST_OUTPUT/checkpoints" ]; then
        CHECKPOINT_COUNT=$(ls "$LATEST_OUTPUT/checkpoints"/*.pth 2>/dev/null | wc -l)
        echo "💾 Checkpoints saved: $CHECKPOINT_COUNT"
        if [ $CHECKPOINT_COUNT -gt 0 ]; then
            echo "📋 Latest checkpoint: $(ls -t "$LATEST_OUTPUT/checkpoints"/*.pth 2>/dev/null | head -1)"
        fi
    fi
    
    # Check metrics file
    if [ -f "$LATEST_OUTPUT/training_metrics.json" ]; then
        echo "📈 Metrics file exists: $LATEST_OUTPUT/training_metrics.json"
        # Try to extract current epoch info
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
try:
    with open('$LATEST_OUTPUT/training_metrics.json', 'r') as f:
        data = json.load(f)
    if 'train' in data and len(data['train']) > 0:
        current_epoch = len(data['train'])
        latest_train = data['train'][-1]
        latest_val = data['val'][-1] if 'val' in data and len(data['val']) > 0 else {}
        print(f'📊 Current epoch: {current_epoch}')
        print(f'📉 Latest train loss: {latest_train.get(\"loss\", \"N/A\"):.4f}' if isinstance(latest_train.get('loss'), (int, float)) else '')
        print(f'📉 Latest val loss: {latest_val.get(\"loss\", \"N/A\"):.4f}' if isinstance(latest_val.get('loss'), (int, float)) else '')
        print(f'🎯 Latest train acc: {latest_train.get(\"accuracy\", \"N/A\"):.4f}' if isinstance(latest_train.get('accuracy'), (int, float)) else '')
        print(f'🎯 Latest val acc: {latest_val.get(\"accuracy\", \"N/A\"):.4f}' if isinstance(latest_val.get('accuracy'), (int, float)) else '')
except Exception as e:
    print(f'Could not parse metrics: {e}')
" 2>/dev/null
        fi
    fi
else
    echo "❌ No output directories found"
fi
echo

# Check port usage (for distributed training)
echo "5. Network ports (distributed training):"
if command -v netstat &> /dev/null; then
    netstat -tuln | grep -E "(29500|12345)" || echo "No training ports found"
elif command -v ss &> /dev/null; then
    ss -tuln | grep -E "(29500|12345)" || echo "No training ports found"
else
    echo "❌ No network monitoring tools available"
fi
echo

echo "=== Quick Commands ==="
echo "📋 Monitor logs: tail -f $LATEST_OUTPUT/training_*.log"
echo "🔄 Watch GPU: watch -n 1 nvidia-smi"
echo "⚡ Kill training: kill \$(ps aux | grep torch.distributed.launch | grep -v grep | awk '{print \$2}')"
echo
