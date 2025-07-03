#!/usr/bin/env python3
"""
Minimal Training Script for Server - Python 3.6 Compatible
No f-strings, handles corrupted images
"""

import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm

from config import Config
from data_loader import get_dataloaders
from model import ViTForProstateCancer, CombinedLoss
from train_utils import train_epoch, validate_epoch, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Minimal Training Script')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/Saif/Pratham/ELC/prostate-cancer-grade-assessment', 
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    print('=== Minimal Training Script ===')
    print('Data directory: {}'.format(config.data_dir))
    print('Output directory: {}'.format(config.output_dir))
    print('Batch size: {}'.format(config.batch_size))
    print('Epochs: {}'.format(config.epochs))
    print()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))
    if torch.cuda.is_available():
        print('GPU: {}'.format(torch.cuda.get_device_name(0)))
    print()
    
    # Initialize model, criterion, optimizer
    print('Initializing model...')
    model = ViTForProstateCancer(config).to(device)
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Get data loaders
    print('Loading data...')
    try:
        train_loader, val_loader = get_dataloaders(config)
        print('Data loaded successfully')
        print('Train samples: {}'.format(len(train_loader.dataset)))
        print('Validation samples: {}'.format(len(val_loader.dataset)))
    except Exception as e:
        print('Error loading data: {}'.format(str(e)))
        return
    print()
    
    # Training loop
    train_metrics = []
    val_metrics = []
    best_metric = float('inf')
    
    for epoch in range(config.epochs):
        print('Epoch {}/{}'.format(epoch + 1, config.epochs))
        print('-' * 50)
        
        # Train
        train_metric = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config.epochs)
        train_metrics.append(train_metric)
        
        # Validate
        val_metric = validate_epoch(model, val_loader, criterion, device)
        val_metrics.append(val_metric)
        
        # Update scheduler
        scheduler.step(val_metric['loss'])
        
        # Print metrics
        print('Train - Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(
            train_metric['loss'], train_metric['accuracy'], train_metric['f1']))
        print('Val   - Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}'.format(
            val_metric['loss'], val_metric['accuracy'], val_metric['f1']))
        
        # Save best model
        if val_metric['loss'] < best_metric:
            best_metric = val_metric['loss']
            best_path = os.path.join(config.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metric, best_path)
            print('New best model saved! Loss: {:.4f}'.format(best_metric))
        
        # Save metrics
        metrics_file = os.path.join(config.output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump({'train': train_metrics, 'val': val_metrics}, f, indent=2)
        
        print()
    
    print('Training completed!')
    print('Best validation loss: {:.4f}'.format(best_metric))

if __name__ == '__main__':
    main()
