"""
Simple configuration for prostate cancer detection training
"""
import os

class Config:
    def __init__(self):
        # Data paths
        self.data_dir = "/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment"
        self.train_csv = "train.csv"
        self.image_dir = "train_images"
        
        # Model settings
        self.model_name = "google/vit-base-patch16-224"
        self.num_classes = 6  # Prostate cancer grades 0-5
        self.image_size = 224
        
        # Training settings
        self.batch_size = 16
        self.epochs = 10
        self.learning_rate = 5e-5
        self.weight_decay = 1e-4
        
        # Output settings
        self.output_dir = "./outputs"
        self.model_save_path = "./outputs/best_model.pth"
        
        # Data loader settings
        self.num_workers = 4
        self.train_split = 0.8
        self.val_split = 0.2
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Check if we have a working model file
        if os.path.exists('working_model.txt'):
            with open('working_model.txt', 'r') as f:
                working_model = f.read().strip()
                print(f"Using pre-downloaded model: {working_model}")
                self.model_name = working_model
