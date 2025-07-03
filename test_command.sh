#!/bin/bash

# Test Training Command
echo "=== Testing Training Command ==="

# Same configuration as start_training.sh
DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
OUTPUT_DIR="./test_outputs"
BATCH_SIZE=2  # Small for testing
EPOCHS=1      # Just 1 epoch for testing
LEARNING_RATE=5e-5

# Create test output directory
mkdir -p "$OUTPUT_DIR"

echo "Testing with these parameters:"
echo "  DATA_DIR: $DATA_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  EPOCHS: $EPOCHS"

# Build the exact command
CMD="python train.py --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --model-save-path $OUTPUT_DIR/test_model.pth --batch-size $BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE"

echo
echo "Command to be executed:"
echo "$CMD"

echo
echo "Testing command execution (dry run)..."

# Test just the argument parsing first
python -c "
import sys
import os

# Set up arguments exactly as they would be passed
sys.argv = [
    'train.py',
    '--data-dir', '$DATA_DIR',
    '--output-dir', '$OUTPUT_DIR', 
    '--model-save-path', '$OUTPUT_DIR/test_model.pth',
    '--batch-size', '$BATCH_SIZE',
    '--epochs', '$EPOCHS',
    '--lr', '$LEARNING_RATE'
]

print('Arguments being passed:')
for i, arg in enumerate(sys.argv):
    print(f'  {i}: {arg}')

print()
try:
    from train import parse_args
    args = parse_args()
    print('✅ Argument parsing successful')
    print(f'   data_dir: {args.data_dir}')
    print(f'   output_dir: {args.output_dir}')
    print(f'   batch_size: {args.batch_size}')
    print(f'   epochs: {args.epochs}')
    
    # Test if data directory is accessible
    if os.path.exists(args.data_dir):
        print(f'✅ Data directory accessible: {args.data_dir}')
        
        # Test CSV file
        import pandas as pd
        csv_path = os.path.join(args.data_dir, 'train.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f'✅ CSV file readable: {df.shape[0]} rows, {df.shape[1]} columns')
        else:
            print(f'❌ CSV file not found: {csv_path}')
    else:
        print(f'❌ Data directory not accessible: {args.data_dir}')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo
echo "If the test above passed, you can run the actual training with:"
echo "  $CMD"

# Clean up test directory
rm -rf "$OUTPUT_DIR"
