#!/bin/bash

# Image File Diagnostic Script
echo "=== Image File Diagnostic ==="
echo "Checking image files in the dataset..."
echo

DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
IMAGE_DIR="$DATA_DIR/train_images"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image directory not found: $IMAGE_DIR"
    exit 1
fi

echo "📁 Image directory: $IMAGE_DIR"
echo "📊 Total files: $(ls "$IMAGE_DIR" | wc -l)"
echo

# Check the problematic file
PROBLEM_FILE="$IMAGE_DIR/376a5058f8fcf38a3df4a0c431e974fa.tiff"
echo "🔍 Checking problematic file: $PROBLEM_FILE"

if [ -f "$PROBLEM_FILE" ]; then
    echo "✅ File exists"
    echo "📏 File size: $(ls -lh "$PROBLEM_FILE" | awk '{print $5}')"
    echo "🗂️  File type: $(file "$PROBLEM_FILE")"
    
    # Check if file is empty or very small
    FILE_SIZE=$(stat -c%s "$PROBLEM_FILE" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000 ]; then
        echo "⚠️  File is very small ($FILE_SIZE bytes) - likely corrupted"
    fi
else
    echo "❌ File not found"
fi

echo
echo "🔍 Checking first 10 image files..."
COUNT=0
GOOD_COUNT=0
BAD_COUNT=0

# Test loading images with Python
python -c "
import os
import sys
from PIL import Image
import traceback

image_dir = '$IMAGE_DIR'
problem_files = []
good_files = []

# Get first 20 files for testing
files = os.listdir(image_dir)[:20]
print('Testing {} image files...'.format(len(files)))
print()

for i, filename in enumerate(files):
    filepath = os.path.join(image_dir, filename)
    try:
        # Try to open and convert to RGB
        with Image.open(filepath) as img:
            img_rgb = img.convert('RGB')
            print('✅ {:2d}. {}: {} {}'.format(i+1, filename, img.size, img.mode))
            good_files.append(filename)
    except Exception as e:
        print('❌ {:2d}. {}: {}'.format(i+1, filename, str(e)))
        problem_files.append((filename, str(e)))

print()
print('Summary: {} good files, {} problem files'.format(len(good_files), len(problem_files)))

if problem_files:
    print()
    print('Problem files:')
    for filename, error in problem_files:
        print('  {}: {}'.format(filename, error))
        
    # Check file details for problem files
    print()
    print('Checking problem file details:')
    for filename, error in problem_files:
        filepath = os.path.join(image_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print('  {}: {} bytes'.format(filename, size))
        else:
            print('  {}: file not found'.format(filename))
"

echo
echo "🔍 Checking file extensions and formats..."
echo "File extension summary:"
ls "$IMAGE_DIR" | sed 's/.*\.//' | sort | uniq -c | sort -nr

echo
echo "🔍 Looking for very small files (likely corrupted)..."
find "$IMAGE_DIR" -type f -size -1k -exec ls -lh {} \; | head -10

echo
echo "🔍 Looking for zero-size files..."
find "$IMAGE_DIR" -type f -size 0 -exec ls -lh {} \;

echo
echo "=== Recommendations ==="
echo
echo "If corrupted files found:"
echo "1. Skip corrupted files during training"
echo "2. Remove corrupted files from CSV"
echo "3. Use try-catch in data loader"
echo
echo "Next steps:"
echo "1. Run: ./fix_dataset.sh (to clean the dataset)"
echo "2. Or: ./start_training_robust.sh (training with error handling)"
