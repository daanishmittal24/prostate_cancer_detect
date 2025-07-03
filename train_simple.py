#!/usr/bin/env python3
"""
Minimal Training Script for Prostate Cancer Detection
Python 3.6+ compatible, handles corrupted images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms

class SimpleConfig:
    def __init__(self):
        self.data_dir = "./prostate-cancer-grade-assessment"
        self.train_csv = "train.csv"
        self.image_dir = "train_images"
        self.img_size = 224
        self.num_classes = 6
        self.batch_size = 16
        self.epochs = 10
        self.learning_rate = 5e-5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4

class SimpleViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            print("Loading ViT model...")
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
            print("✅ Model loaded successfully")
        except Exception as e:
            print("❌ Failed to load pre-trained model: {}".format(e))
            print("Creating model from scratch...")
            vit_config = ViTConfig(
                image_size=config.img_size,
                patch_size=16,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12
            )
            self.vit = ViTModel(vit_config)
            print("✅ Model created from scratch")
        
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, config.num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

class SimpleDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_dir, csv_file))
        print("Loaded {} samples from {}".format(len(self.df), csv_file))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = int(row['isup_grade'])
        
        # Load image with error handling
        image_path = os.path.join(self.data_dir, "train_images", "{}.tiff".format(image_id))
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print("⚠️ Error loading image {}: {}".format(image_id, e))
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': '{:.4f}'.format(total_loss / len(dataloader)),
            'Acc': '{:.2f}%'.format(accuracy)
        })
    
    return total_loss / len(dataloader), accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': '{:.4f}'.format(total_loss / len(dataloader)),
                'Acc': '{:.2f}%'.format(accuracy)
            })
    
    return total_loss / len(dataloader), accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='./outputs_simple', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()
    
    # Configuration
    config = SimpleConfig()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    print("=== Simple Training Configuration ===")
    print("Data directory: {}".format(config.data_dir))
    print("Output directory: {}".format(args.output_dir))
    print("Batch size: {}".format(config.batch_size))
    print("Epochs: {}".format(config.epochs))
    print("Learning rate: {}".format(config.learning_rate))
    print("Device: {}".format(config.device))
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    train_transform, val_transform = get_transforms()
    
    # Load dataset
    print("Loading datasets...")
    full_dataset = SimpleDataset(config.train_csv, config.data_dir, train_transform)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Data loaders
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
    
    print("Train samples: {}".format(len(train_dataset)))
    print("Validation samples: {}".format(len(val_dataset)))
    print()
    
    # Model
    device = torch.device(config.device)
    model = SimpleViTModel(config).to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print("Starting training...")
    for epoch in range(config.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, config.epochs))
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print("\nEpoch {} Summary:".format(epoch + 1))
        print("Train Loss: {:.4f}, Train Acc: {:.2f}%".format(train_loss, train_acc))
        print("Val Loss: {:.4f}, Val Acc: {:.2f}%".format(val_loss, val_acc))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            print("✅ New best model saved! Val Acc: {:.2f}%".format(val_acc))
        
        # Save metrics
        metrics = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
        
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("Best validation accuracy: {:.2f}%".format(best_val_acc))
    print("Model saved to: {}".format(os.path.join(args.output_dir, 'best_model.pth')))

if __name__ == '__main__':
    main()
