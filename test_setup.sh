#!/bin/bash

# Test Script - Run a quick training test
echo "=== Quick Training Test ==="
echo "This script runs a minimal test to verify everything works"
echo

# Test single GPU training first
echo "1. Testing single GPU/CPU training..."

python -c "
import torch
import sys
import os

# Basic checks
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')

# Test imports
try:
    from transformers import ViTModel
    print('✅ transformers import successful')
except ImportError as e:
    print(f'❌ transformers import failed: {e}')
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    print('✅ Data processing imports successful')
except ImportError as e:
    print(f'❌ Data processing imports failed: {e}')
    sys.exit(1)

print('✅ All basic imports successful')
"

if [ $? -ne 0 ]; then
    echo "❌ Basic test failed. Please check your environment."
    exit 1
fi

echo
echo "2. Testing distributed setup (if multiple GPUs available)..."

AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ "$AVAILABLE_GPUS" -gt 1 ]; then
    echo "Testing distributed initialization..."
    python -c "
import torch
import torch.distributed as dist
import os

if torch.cuda.device_count() > 1:
    print(f'Multiple GPUs available: {torch.cuda.device_count()}')
    
    # Test NCCL availability
    if torch.distributed.is_nccl_available():
        print('✅ NCCL backend available')
    else:
        print('⚠️  NCCL not available, gloo might work')
        
    # Test gloo availability  
    if torch.distributed.is_gloo_available():
        print('✅ Gloo backend available')
    else:
        print('❌ Gloo not available')
else:
    print('Single GPU detected, distributed training not needed')
"
else
    echo "Single or no GPU detected, skipping distributed test"
fi

echo
echo "3. Testing data loading (with dummy data)..."

# Create a small test dataset
TEST_DIR="./test_data"
mkdir -p "$TEST_DIR/train_images"
mkdir -p "$TEST_DIR/train_label_masks"

# Create dummy CSV
cat > "$TEST_DIR/train.csv" << EOF
image_id,data_provider,isup_grade,gleason_score
dummy_001,provider1,1,6
dummy_002,provider1,2,7
dummy_003,provider1,0,0
EOF

# Create dummy images (small black images)
python -c "
from PIL import Image
import numpy as np
import os

test_dir = './test_data/train_images'
os.makedirs(test_dir, exist_ok=True)

for i in range(3):
    # Create a small black image
    img = Image.new('RGB', (64, 64), color='black')
    img.save(f'{test_dir}/dummy_00{i+1}.tiff')
    print(f'Created dummy_00{i+1}.tiff')

print('✅ Dummy test data created')
"

# Test data loading
python -c "
import sys
sys.path.append('.')

try:
    from config import Config
    from data_loader import get_dataloaders
    
    # Create test config
    config = Config()
    config.data_dir = './test_data'
    config.batch_size = 2
    config.img_size = 64
    config.num_workers = 0  # Avoid multiprocessing issues in test
    
    # Test data loading
    train_loader, val_loader = get_dataloaders(config)
    print(f'✅ Data loading successful')
    print(f'   Train batches: {len(train_loader)}')
    print(f'   Val batches: {len(val_loader)}')
    
    # Test one batch
    for batch in train_loader:
        if len(batch) == 3:
            images, masks, labels = batch
            print(f'   Batch shape: images={images.shape}, masks={masks.shape}, labels={labels.shape}')
        else:
            images, labels = batch
            print(f'   Batch shape: images={images.shape}, labels={labels.shape}')
        break
        
except Exception as e:
    print(f'❌ Data loading test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Clean up test data
rm -rf "$TEST_DIR"

echo
echo "4. Testing model creation..."

python -c "
import sys
sys.path.append('.')

try:
    from config import Config
    from model import ViTForProstateCancer, CombinedLoss
    import torch
    
    config = Config()
    config.img_size = 64  # Small for test
    
    # Test model creation
    model = ViTForProstateCancer(config)
    criterion = CombinedLoss(alpha=0.7)
    
    print('✅ Model creation successful')
    print(f'   Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits, masks = model(dummy_input)
        print(f'   Forward pass successful: logits={logits.shape}, masks={masks.shape}')
        
except Exception as e:
    print(f'❌ Model test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo
echo "✅ All tests passed! Your setup should work for training."
echo
echo "Next steps:"
echo "1. Update the DATA_DIR path in run_training_improved.sh"
echo "2. Run: ./run_training_improved.sh"
echo "3. Monitor with: ./monitor_training.sh"
