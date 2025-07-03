#!/bin/bash

# Quick Fix and Start Training
echo "=== Quick Fix and Start Training ==="
echo "Date: $(date)"
echo

# Step 1: Check images with fixed script
echo "🔍 Step 1: Checking for corrupted images..."
./check_images.sh

echo
echo "🚀 Step 2: Starting robust training..."
echo "This will handle any corrupted images automatically."
echo

# Step 2: Start robust training
./start_training_robust.sh

echo
echo "✅ Training started!"
echo "Use ./monitor.sh to check progress"
