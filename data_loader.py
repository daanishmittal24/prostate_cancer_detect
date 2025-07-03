"""
Simple data loader for prostate cancer detection
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class ProstateDataset(Dataset):
    def __init__(self, csv_file, data_dir, image_dir, transform=None):
        self.df = pd.read_csv(os.path.join(data_dir, csv_file))
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['isup_grade']
        
        # Load image
        image_path = os.path.join(self.data_dir, self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(config):
    """Create train and validation data loaders"""
    # Read the full dataset
    csv_path = os.path.join(config.data_dir, config.train_csv)
    df = pd.read_csv(csv_path)
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, test_size=config.val_split, random_state=42, stratify=df['isup_grade']
    )
    
    # Save split dataframes
    train_df.to_csv(os.path.join(config.output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(config.output_dir, 'val_split.csv'), index=False)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = ProstateDataset(
        csv_file='train_split.csv',
        data_dir=config.output_dir,
        image_dir=os.path.join(config.data_dir, config.image_dir),
        transform=train_transform
    )
    
    val_dataset = ProstateDataset(
        csv_file='val_split.csv', 
        data_dir=config.output_dir,
        image_dir=os.path.join(config.data_dir, config.image_dir),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader
    )
    
    # Define transforms
    train_transforms = get_transforms(config.img_size, is_train=True)
    val_transforms = get_transforms(config.img_size, is_train=False)
    
    # Create datasets first
    train_dataset = ProstateDataset(
        train_df,
        os.path.join(config.data_dir, config.image_dir),
        os.path.join(config.data_dir, config.mask_dir) if config.mask_dir else None,
        transform=train_transforms
    )
    
    val_dataset = ProstateDataset(
        val_df,
        os.path.join(config.data_dir, config.image_dir),
        os.path.join(config.data_dir, config.mask_dir) if config.mask_dir else None,
        transform=val_transforms
    )
    
    # For distributed training, we need to handle the dataset splitting differently
    if config.distributed:
        # In distributed mode, we'll use DistributedSampler
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False  # Shuffling is handled by the sampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True  # Helps with batch norm in distributed training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Don't need to shuffle validation data
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )
    
    return train_loader, val_loader
