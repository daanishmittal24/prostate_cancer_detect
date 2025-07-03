import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class ProstateDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir=None, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx]['image_id'] + '.tiff')
        
        # Load image with error handling for different TIFF formats
        image = self._load_image_robust(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.df.iloc[idx]['image_id']
            
        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.df.iloc[idx]['image_id'] + '_mask.tiff')
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                if self.transform:
                    mask = self.transform(mask)
            
        # Get label (ISUP grade)
        label = torch.tensor(self.df.iloc[idx]['isup_grade'], dtype=torch.long)
        
        if mask is not None:
            return image, mask, label
        return image, label

    def _load_image_robust(self, img_path):
        """
        Robust image loading that tries multiple methods for TIFF files
        """
        # Method 1: Try PIL first (fastest)
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            print(f"PIL failed for {img_path}: {e}")
        
        # Method 2: Try OpenCV
        try:
            import cv2
            img_cv2 = cv2.imread(img_path)
            if img_cv2 is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_rgb)
                return image
            else:
                print(f"OpenCV returned None for {img_path}")
        except ImportError:
            print("OpenCV not available")
        except Exception as e:
            print(f"OpenCV failed for {img_path}: {e}")
        
        # Method 3: Try tifffile
        try:
            import tifffile
            import numpy as np
            img_array = tifffile.imread(img_path)
            
            # Handle different array shapes and types
            if img_array.ndim == 2:
                # Grayscale - convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim == 3 and img_array.shape[0] == 3:
                # Channels first - transpose to channels last
                img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(img_array)
            return image
            
        except ImportError:
            print("tifffile not available")
        except Exception as e:
            print(f"tifffile failed for {img_path}: {e}")
        
        # Method 4: Try scikit-image
        try:
            from skimage import io
            import numpy as np
            img_array = io.imread(img_path)
            
            # Normalize and convert if needed
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
            
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            image = Image.fromarray(img_array)
            return image
            
        except ImportError:
            print("scikit-image not available")
        except Exception as e:
            print(f"scikit-image failed for {img_path}: {e}")
        
        # If all methods fail, create a dummy image
        print(f"❌ All image loading methods failed for {img_path}")
        print("Creating dummy black image...")
        dummy_image = Image.new('RGB', (224, 224), color='black')
        return dummy_image

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

def get_dataloaders(config):
    # Load data
    df = pd.read_csv(os.path.join(config.data_dir, config.train_csv))
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['isup_grade']
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
