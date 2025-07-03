#!/bin/bash

# TIFF Image Diagnostic Script
echo "=== TIFF Image Diagnostic ==="
echo "Checking TIFF file issues..."
echo

DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
PROBLEM_FILE="$DATA_DIR/train_images/376a5058f8fcf38a3df4a0c431e974fa.tiff"

echo "1. Checking the problematic file..."
if [ -f "$PROBLEM_FILE" ]; then
    echo "✅ File exists: $PROBLEM_FILE"
    echo "   File size: $(ls -lh "$PROBLEM_FILE" | awk '{print $5}')"
    echo "   File type: $(file "$PROBLEM_FILE")"
    echo "   Permissions: $(ls -l "$PROBLEM_FILE" | awk '{print $1}')"
else
    echo "❌ File not found: $PROBLEM_FILE"
fi

echo
echo "2. Checking sample of other TIFF files..."
SAMPLE_FILES=$(ls "$DATA_DIR/train_images/"*.tiff 2>/dev/null | head -5)
if [ -n "$SAMPLE_FILES" ]; then
    echo "Sample TIFF files:"
    for file in $SAMPLE_FILES; do
        echo "   $(basename "$file"): $(file "$file" | cut -d: -f2)"
    done
else
    echo "❌ No TIFF files found"
fi

echo
echo "3. Testing Python image loading..."
python -c "
import os
from PIL import Image
import numpy as np

data_dir = '/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment'
img_dir = os.path.join(data_dir, 'train_images')

# Test the problematic file
problem_file = '/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment/train_images/376a5058f8fcf38a3df4a0c431e974fa.tiff'

print('Testing problematic file...')
try:
    img = Image.open(problem_file)
    print(f'✅ PIL can open: {img.size}, {img.mode}')
    rgb_img = img.convert('RGB')
    print(f'✅ RGB conversion successful: {rgb_img.size}')
except Exception as e:
    print(f'❌ PIL failed: {e}')
    
    # Try alternative methods
    try:
        import cv2
        img_cv2 = cv2.imread(problem_file)
        if img_cv2 is not None:
            print(f'✅ OpenCV can read: {img_cv2.shape}')
        else:
            print('❌ OpenCV also failed')
    except ImportError:
        print('OpenCV not available for testing')
    
    try:
        import tifffile
        img_tiff = tifffile.imread(problem_file)
        print(f'✅ tifffile can read: {img_tiff.shape}')
    except ImportError:
        print('tifffile not available for testing')
    except Exception as e2:
        print(f'❌ tifffile failed: {e2}')

# Test a few other files
print()
print('Testing other TIFF files...')
tiff_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')][:5]

working_files = 0
for i, filename in enumerate(tiff_files):
    filepath = os.path.join(img_dir, filename)
    try:
        img = Image.open(filepath)
        img_rgb = img.convert('RGB')
        working_files += 1
        if i == 0:  # Show details for first working file
            print(f'✅ {filename}: {img.size}, {img.mode}')
    except Exception as e:
        print(f'❌ {filename}: {e}')

print(f)
print(f'Summary: {working_files}/{len(tiff_files)} files can be loaded with PIL')
"

echo
echo "4. Checking available image libraries..."
python -c "
try:
    from PIL import Image
    print('✅ PIL/Pillow available')
except ImportError:
    print('❌ PIL/Pillow not available')

try:
    import cv2
    print('✅ OpenCV available')
except ImportError:
    print('❌ OpenCV not available')

try:
    import tifffile
    print('✅ tifffile available')
except ImportError:
    print('❌ tifffile not available')

try:
    import skimage
    print('✅ scikit-image available')
except ImportError:
    print('❌ scikit-image not available')
"

echo
echo "5. Recommendations..."
echo "If many files fail to load with PIL, try:"
echo "1. Install additional image libraries: pip install tifffile scikit-image"
echo "2. Update PIL/Pillow: pip install --upgrade Pillow"
echo "3. Use alternative loading in data_loader.py"
