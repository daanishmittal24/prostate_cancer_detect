#!/bin/bash

# Data Path Fix Script
echo "=== Data Path Diagnostic and Fix ==="
echo "Current working directory: $(pwd)"
echo

# Test the data path
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
echo "Testing data directory: $DATA_DIR"

if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory exists"
    echo "Contents:"
    ls -la "$DATA_DIR"
    echo
    
    # Check specific files
    if [ -f "$DATA_DIR/train.csv" ]; then
        echo "✅ train.csv found"
        echo "   File size: $(ls -lh "$DATA_DIR/train.csv" | awk '{print $5}')"
        echo "   First few lines:"
        head -3 "$DATA_DIR/train.csv"
    else
        echo "❌ train.csv not found"
        echo "Available CSV files:"
        find "$DATA_DIR" -name "*.csv" -type f
    fi
    
    echo
    if [ -d "$DATA_DIR/train_images" ]; then
        echo "✅ train_images directory found"
        IMAGE_COUNT=$(ls "$DATA_DIR/train_images" | wc -l)
        echo "   Image count: $IMAGE_COUNT"
        echo "   Sample files:"
        ls "$DATA_DIR/train_images" | head -3
    else
        echo "❌ train_images directory not found"
        echo "Available directories:"
        find "$DATA_DIR" -type d -name "*image*"
    fi
    
    echo
    if [ -d "$DATA_DIR/train_label_masks" ]; then
        echo "✅ train_label_masks directory found"
        MASK_COUNT=$(ls "$DATA_DIR/train_label_masks" | wc -l)
        echo "   Mask count: $MASK_COUNT"
    else
        echo "❌ train_label_masks directory not found"
        echo "Available mask directories:"
        find "$DATA_DIR" -type d -name "*mask*"
    fi
    
else
    echo "❌ Data directory not found: $DATA_DIR"
    echo
    echo "Searching for prostate cancer data..."
    find /home/Saif/Pratham/ELC/ -name "*prostate*" -type d 2>/dev/null
    echo
    echo "Searching for train.csv files..."
    find /home/Saif/Pratham/ELC/ -name "train.csv" -type f 2>/dev/null
fi

echo
echo "=== Testing Python Data Loading ==="
python -c "
import os
import pandas as pd

data_dir = '/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment'
print(f'Testing data directory: {data_dir}')

try:
    # Test CSV loading
    csv_path = os.path.join(data_dir, 'train.csv')
    print(f'Trying to load: {csv_path}')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f'✅ CSV loaded successfully')
        print(f'   Shape: {df.shape}')
        print(f'   Columns: {list(df.columns)}')
        if 'isup_grade' in df.columns:
            print(f'   ISUP grades: {sorted(df.isup_grade.unique())}')
    else:
        print(f'❌ CSV file not found: {csv_path}')
        
    # Test image directory
    img_dir = os.path.join(data_dir, 'train_images')
    if os.path.exists(img_dir):
        images = os.listdir(img_dir)
        print(f'✅ Images directory accessible: {len(images)} files')
    else:
        print(f'❌ Images directory not found: {img_dir}')
        
except Exception as e:
    print(f'❌ Error: {e}')
"

echo
echo "=== Quick Training Test (Dry Run) ==="
echo "Testing command line argument parsing..."

python -c "
import sys
sys.path.append('.')

# Test argument parsing
test_args = [
    'train.py',
    '--data-dir', '/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment',
    '--output-dir', './test_output',
    '--batch-size', '2',
    '--epochs', '1',
    '--lr', '1e-4'
]

# Temporarily replace sys.argv
original_argv = sys.argv
sys.argv = test_args

try:
    from train import parse_args
    args = parse_args()
    print('✅ Argument parsing successful')
    print(f'   Data dir: {args.data_dir}')
    print(f'   Output dir: {args.output_dir}')
    print(f'   Batch size: {args.batch_size}')
except Exception as e:
    print(f'❌ Argument parsing failed: {e}')
finally:
    sys.argv = original_argv
"

echo
echo "=== Recommendations ==="
echo "If train.csv was found, you can now run:"
echo "  ./start_training.sh"
echo
echo "If there were path issues, check:"
echo "1. Correct data directory path"
echo "2. File permissions"
echo "3. CSV file format"
