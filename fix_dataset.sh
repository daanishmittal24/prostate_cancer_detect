#!/bin/bash

# Dataset Cleaning Script
echo "=== Dataset Cleaning Script ==="
echo "This script will clean the dataset by removing corrupted images"
echo

DATA_DIR="/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
IMAGE_DIR="$DATA_DIR/train_images"
CSV_FILE="$DATA_DIR/train.csv"
BACKUP_DIR="$DATA_DIR/backup_$(date +%Y%m%d_%H%M%S)"

echo "📁 Data directory: $DATA_DIR"
echo "🗂️  CSV file: $CSV_FILE"
echo "💾 Backup directory: $BACKUP_DIR"
echo

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup original CSV
if [ -f "$CSV_FILE" ]; then
    cp "$CSV_FILE" "$BACKUP_DIR/train_original.csv"
    echo "✅ Backed up original CSV"
else
    echo "❌ CSV file not found: $CSV_FILE"
    exit 1
fi

echo
echo "🔍 Finding corrupted images..."

# Create Python script to find and clean corrupted images
python -c "
import os
import pandas as pd
from PIL import Image
import shutil

data_dir = '$DATA_DIR'
image_dir = '$IMAGE_DIR'
csv_file = '$CSV_FILE'
backup_dir = '$BACKUP_DIR'

print('Loading CSV file...')
df = pd.read_csv(csv_file)
print(f'Original dataset: {len(df)} rows')

good_files = []
corrupted_files = []
missing_files = []

print('Checking all images in CSV...')
for idx, row in df.iterrows():
    image_id = row['image_id']
    image_path = os.path.join(image_dir, f'{image_id}.tiff')
    
    if idx % 1000 == 0:
        print(f'Checked {idx}/{len(df)} images...')
    
    if not os.path.exists(image_path):
        missing_files.append(image_id)
        continue
    
    try:
        # Try to open and convert image
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            # Check if image has reasonable size
            if img.size[0] > 10 and img.size[1] > 10:
                good_files.append(image_id)
            else:
                corrupted_files.append(image_id)
    except Exception as e:
        corrupted_files.append(image_id)
        
print(f'Scan complete:')
print(f'  Good files: {len(good_files)}')
print(f'  Corrupted files: {len(corrupted_files)}')
print(f'  Missing files: {len(missing_files)}')

# Create cleaned dataset
if corrupted_files or missing_files:
    bad_files = set(corrupted_files + missing_files)
    clean_df = df[~df['image_id'].isin(bad_files)].copy()
    
    print(f'Creating cleaned dataset: {len(clean_df)} rows (removed {len(df) - len(clean_df)} rows)')
    
    # Save cleaned CSV
    clean_csv_path = os.path.join(data_dir, 'train_clean.csv')
    clean_df.to_csv(clean_csv_path, index=False)
    print(f'Saved cleaned CSV: {clean_csv_path}')
    
    # Save list of problematic files
    if corrupted_files:
        with open(os.path.join(backup_dir, 'corrupted_files.txt'), 'w') as f:
            f.write('\\n'.join(corrupted_files))
        print(f'Saved corrupted files list: {len(corrupted_files)} files')
    
    if missing_files:
        with open(os.path.join(backup_dir, 'missing_files.txt'), 'w') as f:
            f.write('\\n'.join(missing_files))
        print(f'Saved missing files list: {len(missing_files)} files')
        
    # Optionally move corrupted files to backup
    print('Moving corrupted files to backup...')
    corrupted_backup_dir = os.path.join(backup_dir, 'corrupted_images')
    os.makedirs(corrupted_backup_dir, exist_ok=True)
    
    moved_count = 0
    for image_id in corrupted_files:
        src_path = os.path.join(image_dir, f'{image_id}.tiff')
        if os.path.exists(src_path):
            dst_path = os.path.join(corrupted_backup_dir, f'{image_id}.tiff')
            shutil.move(src_path, dst_path)
            moved_count += 1
    
    print(f'Moved {moved_count} corrupted files to backup')
    
    print('Dataset cleaning complete!')
    print(f'Use train_clean.csv for training ({len(clean_df)} samples)')
    
else:
    print('No corrupted or missing files found - dataset is clean!')
"

echo
echo "✅ Dataset cleaning complete!"
echo
echo "Files created:"
echo "  - train_clean.csv (cleaned dataset)"
echo "  - backup/ directory with original files"
echo
echo "Next step: Update config to use cleaned dataset"
