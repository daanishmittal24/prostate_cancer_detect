#!/bin/bash

# Quick Test Script
echo "=== Quick Test - Verify Everything Works ==="

# Test 1: Check local model
echo "1. Testing local ViT model..."
python -c "
import os
from transformers import ViTModel
import torch

if os.path.exists('./local_vit_model'):
    print('✅ Local model directory found')
    try:
        model = ViTModel.from_pretrained('./local_vit_model')
        print('✅ Model loads successfully')
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f'✅ Forward pass works: {output.last_hidden_state.shape}')
    except Exception as e:
        print(f'❌ Model test failed: {e}')
        exit(1)
else:
    print('❌ Local model not found')
    exit(1)
"

# Test 2: Check data access
echo
echo "2. Testing data directory access..."
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory accessible: $DATA_DIR"
    
    # Check key files
    if [ -f "$DATA_DIR/train.csv" ]; then
        ROWS=$(wc -l < "$DATA_DIR/train.csv")
        echo "✅ train.csv found ($ROWS rows)"
    else
        echo "❌ train.csv not found"
        echo "Available files:"
        ls -la "$DATA_DIR" | head -10
    fi
    
    if [ -d "$DATA_DIR/train_images" ]; then
        IMAGES=$(ls "$DATA_DIR/train_images" | wc -l)
        echo "✅ train_images directory found ($IMAGES files)"
    else
        echo "❌ train_images directory not found"
    fi
else
    echo "❌ Data directory not accessible: $DATA_DIR"
    echo "Please check the path or update DATA_DIR in start_training.sh"
fi

# Test 3: Check GPU
echo
echo "3. Testing GPU..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ No GPU available')
"

# Test 4: Quick training test (1 step)
echo
echo "4. Quick training test (single forward/backward pass)..."
python -c "
import torch
import sys
import os
sys.path.append('.')

try:
    from config import Config
    from model import ViTForProstateCancer, CombinedLoss
    
    # Create test config
    config = Config()
    config.model_name = './local_vit_model'  # Use local model
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTForProstateCancer(config).to(device)
    criterion = CombinedLoss(alpha=0.7)
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 6, (2,)).to(device)
    
    logits, masks = model(dummy_images)
    loss, cls_loss, seg_loss = criterion(logits, masks, dummy_labels, None)
    
    print(f'✅ Model forward pass successful')
    print(f'   Logits shape: {logits.shape}')
    print(f'   Masks shape: {masks.shape}')
    print(f'   Loss: {loss.item():.4f}')
    
    # Test backward pass
    loss.backward()
    print(f'✅ Backward pass successful')
    
except Exception as e:
    print(f'❌ Training test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo
echo "✅ All tests passed! Ready to start training."
echo
echo "To start training:"
echo "  ./start_training.sh"
echo
echo "To monitor training:"
echo "  ./monitor.sh"
