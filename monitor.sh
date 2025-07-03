#!/bin/bash

# Training Monitor for SSH Server
echo "=== Training Monitor Dashboard ==="
echo "Server: $(hostname)"
echo "Date: $(date)"
echo

# Find the latest output directory
LATEST_OUTPUT=$(ls -td outputs_* 2>/dev/null | head -1)

if [ -z "$LATEST_OUTPUT" ]; then
    echo "❌ No training outputs found"
    echo "Have you started training yet? Run: ./start_training.sh"
    exit 1
fi

echo "📁 Latest training: $LATEST_OUTPUT"
echo

# Check if training process is running
echo "1. Process Status:"
if [ -f "$LATEST_OUTPUT/train.pid" ]; then
    TRAIN_PID=$(cat "$LATEST_OUTPUT/train.pid")
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "✅ Training is running (PID: $TRAIN_PID)"
        PROCESS_INFO=$(ps -p $TRAIN_PID -o pid,ppid,etime,pcpu,pmem,cmd --no-headers)
        echo "   $PROCESS_INFO"
    else
        echo "❌ Training process not found (PID: $TRAIN_PID)"
        echo "   Process may have finished or crashed"
    fi
else
    echo "⚠️  No PID file found"
    # Check for any python training processes
    PYTHON_PROCS=$(ps aux | grep "python.*train.py" | grep -v grep)
    if [ -n "$PYTHON_PROCS" ]; then
        echo "Found training processes:"
        echo "$PYTHON_PROCS"
    else
        echo "No training processes found"
    fi
fi

echo

# GPU Status
echo "2. GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo

# Training Progress
echo "3. Training Progress:"
LOG_FILE=$(ls -t "$LATEST_OUTPUT"/training_*.log 2>/dev/null | head -1)

if [ -f "$LOG_FILE" ]; then
    echo "📄 Log file: $LOG_FILE"
    echo "📏 Log size: $(du -h "$LOG_FILE" | cut -f1)"
    echo "⏰ Last modified: $(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)"
    
    echo
    echo "4. Recent Log Output (last 15 lines):"
    echo "----------------------------------------"
    tail -15 "$LOG_FILE"
    echo "----------------------------------------"
    
    echo
    echo "5. Training Metrics:"
    # Try to extract epoch information
    EPOCHS_COMPLETED=$(grep -c "Epoch.*Training Progress" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "   Epochs completed: $EPOCHS_COMPLETED"
    
    # Look for error messages
    ERRORS=$(grep -i "error\|exception\|failed" "$LOG_FILE" | tail -3)
    if [ -n "$ERRORS" ]; then
        echo "   ⚠️  Recent errors found:"
        echo "$ERRORS"
    fi
    
    # Look for best model saves
    BEST_SAVES=$(grep -c "New best model saved" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "   Best model saves: $BEST_SAVES"
    
else
    echo "❌ No log file found"
fi

echo

# Check output files
echo "6. Output Files:"
if [ -d "$LATEST_OUTPUT/checkpoints" ]; then
    CHECKPOINT_COUNT=$(ls "$LATEST_OUTPUT/checkpoints"/*.pth 2>/dev/null | wc -l)
    echo "   Checkpoints saved: $CHECKPOINT_COUNT"
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "   Latest checkpoint: $(ls -t "$LATEST_OUTPUT/checkpoints"/*.pth 2>/dev/null | head -1)"
    fi
else
    echo "   No checkpoints directory found yet"
fi

if [ -f "$LATEST_OUTPUT/training_metrics.json" ]; then
    echo "   ✅ Metrics file exists"
else
    echo "   ⏳ Metrics file not created yet"
fi

echo

# Quick commands
echo "=== Quick Commands ==="
echo "📊 View live logs: tail -f \"$LOG_FILE\""
echo "🔄 Watch GPU: watch -n 1 nvidia-smi"
echo "⏹️  Stop training: kill \$(cat \"$LATEST_OUTPUT/train.pid\")"
echo "📁 Check outputs: ls -la \"$LATEST_OUTPUT\""

# Show live logs option
echo
echo "📺 Would you like to view live logs now? (y/N)"
read -t 5 -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Starting live log view (Ctrl+C to exit)..."
    sleep 1
    tail -f "$LOG_FILE"
fi
