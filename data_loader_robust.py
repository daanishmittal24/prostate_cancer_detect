import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustProstateDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir=None, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        self.failed_files = set()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.df)
                image_id = self.df.iloc[current_idx]['image_id']
                
                # Skip if we know this file is corrupted
                if image_id in self.failed_files:
                    continue
                    
                img_name = os.path.join(self.image_dir, image_id + '.tiff')
                
                # Check if file exists
                if not os.path.exists(img_name):
                    logger.warning(f"Image file not found: {img_name}")
                    self.failed_files.add(image_id)
                    continue
                
                # Check file size
                if os.path.getsize(img_name) < 1000:  # Less than 1KB
                    logger.warning(f"Image file too small: {img_name}")
                    self.failed_files.add(image_id)
                    continue
                
                # Try multiple methods to load the image
                image = None
                
                # Method 1: PIL
                try:
                    image = Image.open(img_name).convert('RGB')
                except Exception as e:
                    logger.debug(f"PIL failed for {img_name}: {e}")
                
                # Method 2: OpenCV if PIL fails
                if image is None:
                    try:
                        img_cv = cv2.imread(img_name)
                        if img_cv is not None:
                            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(img_cv)
                    except Exception as e:
                        logger.debug(f"OpenCV failed for {img_name}: {e}")
                
                # Method 3: Create a dummy image if all else fails
                if image is None:
                    logger.warning(f"Creating dummy image for corrupted file: {img_name}")
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    self.failed_files.add(image_id)
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                if self.is_test:
                    return image, image_id
                    
                # Load mask if available
                mask = None
                if self.mask_dir:
                    mask_path = os.path.join(self.mask_dir, image_id + '_mask.tiff')
                    if os.path.exists(mask_path):
                        try:
                            mask = Image.open(mask_path).convert('L')
                            if self.transform:
                                mask = self.transform(mask)
                        except Exception as e:
                            logger.debug(f"Mask loading failed for {mask_path}: {e}")
                            mask = None
                
                # Get label (ISUP grade)
                label = torch.tensor(self.df.iloc[current_idx]['isup_grade'], dtype=torch.long)
                
                if mask is not None:
                    return image, mask, label
                return image, label
                
            except Exception as e:
                logger.warning(f"Error loading sample {idx} (retry {retry}): {e}")
                if retry == max_retries - 1:
                    # Last resort: return a dummy sample
                    logger.error(f"Failed to load any valid sample around index {idx}, returning dummy")
                    dummy_image = torch.zeros(3, 224, 224)
                    dummy_label = torch.tensor(0, dtype=torch.long)
                    return dummy_image, dummy_label
                continue
        
        # Should not reach here, but just in case
        dummy_image = torch.zeros(3, 224, 224)
        dummy_label = torch.tensor(0, dtype=torch.long)
        return dummy_image, dummy_label

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders_robust(config, train_csv_name=None):
    """
    Robust data loader that handles corrupted images
    """
    # Use custom CSV name if provided
    csv_name = train_csv_name if train_csv_name else config.train_csv
    csv_path = os.path.join(config.data_dir, csv_name)
    
    logger.info(f"Loading dataset from: {csv_path}")
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise
    
    # Check for required columns
    required_columns = ['image_id', 'isup_grade']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['isup_grade']
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    
    # For distributed training, we need to handle the dataset splitting differently
    if config.distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_df, shuffle=True)
        val_sampler = DistributedSampler(val_df, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # Define transforms
    train_transforms = get_transforms(config.img_size, is_train=True)
    val_transforms = get_transforms(config.img_size, is_train=False)
    
    # Create robust datasets
    train_dataset = RobustProstateDataset(
        train_df.reset_index(drop=True),
        os.path.join(config.data_dir, config.image_dir),
        os.path.join(config.data_dir, config.mask_dir) if config.mask_dir else None,
        transform=train_transforms
    )
    
    val_dataset = RobustProstateDataset(
        val_df.reset_index(drop=True),
        os.path.join(config.data_dir, config.image_dir),
        os.path.join(config.data_dir, config.mask_dir) if config.mask_dir else None,
        transform=val_transforms
    )
    
    # Create dataloaders with error handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=min(config.num_workers, 2),  # Reduce workers to avoid issues
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=False  # Avoid worker issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader
