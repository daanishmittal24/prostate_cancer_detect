#!/bin/bash

# Stop Training Script
# This script helps you safely stop the training process

echo "=== Stop Training Script ==="
echo "Date: $(date)"
echo

# Find training processes
echo "1. Looking for training processes..."
TRAIN_PIDS=$(ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep | awk '{print $2}')

if [ -z "$TRAIN_PIDS" ]; then
    echo "❌ No training processes found"
    exit 0
fi

echo "✅ Found training processes with PIDs: $TRAIN_PIDS"
echo

# Show process details
echo "2. Process details:"
ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep
echo

# Ask for confirmation
echo "3. Do you want to stop these training processes? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Stopping training processes..."
    
    # First try graceful shutdown (SIGTERM)
    for pid in $TRAIN_PIDS; do
        echo "Sending SIGTERM to PID $pid..."
        kill -TERM "$pid" 2>/dev/null || echo "Could not send SIGTERM to $pid"
    done
    
    # Wait a bit for graceful shutdown
    echo "Waiting 10 seconds for graceful shutdown..."
    sleep 10
    
    # Check if processes are still running
    REMAINING_PIDS=$(ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep | awk '{print $2}')
    
    if [ -n "$REMAINING_PIDS" ]; then
        echo "Some processes are still running. Sending SIGKILL..."
        for pid in $REMAINING_PIDS; do
            echo "Sending SIGKILL to PID $pid..."
            kill -KILL "$pid" 2>/dev/null || echo "Could not send SIGKILL to $pid"
        done
    fi
    
    # Final check
    sleep 2
    FINAL_CHECK=$(ps aux | grep -E "(torch.distributed.launch|train.py)" | grep -v grep | awk '{print $2}')
    
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ All training processes stopped successfully"
    else
        echo "⚠️  Some processes may still be running: $FINAL_CHECK"
        echo "You may need to kill them manually with: kill -9 $FINAL_CHECK"
    fi
    
    echo
    echo "4. Final GPU check:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    fi
    
else
    echo "Training processes left running."
fi

echo "Done."
