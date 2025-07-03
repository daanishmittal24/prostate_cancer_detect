import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    data_dir: str = "./prostate-cancer-grade-assessment"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    image_dir: str = "train_images"
    mask_dir: str = "train_label_masks"
    
    # Model
    model_name: str = "google/vit-base-patch16-224-in21k"
    img_size: int = 224
    num_classes: int = 6  # ISUP grades 0-5
    
    # Training
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 5e-5
    patience: int = 5
    
    # Paths
    output_dir: str = "./outputs"
    model_save_path: str = "./models/vit_prostate"
    
    # Data augmentation
    h_flip_prob: float = 0.5
    v_flip_prob: float = 0.5
    rotation_range: int = 10
    brightness_range: tuple = (0.8, 1.2)
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    rank: int = 0
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
