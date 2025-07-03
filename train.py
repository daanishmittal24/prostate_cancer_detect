import os
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import argparse

from config import Config
from data_loader import get_dataloaders
from model import ViTForProstateCancer, CombinedLoss
from train_utils import train_epoch, validate_epoch, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Train Prostate Cancer Detection Model')
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Path to save outputs')
    parser.add_argument('--model-save-path', type=str, default='./models/vit_prostate.pth', 
                        help='Path to save the best model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                       help='distributed backend')
    parser.add_argument('--dist-url', default='env://', type=str,
                       help='url used to set up distributed training')
    
    return parser.parse_args()

def setup_environment():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_metrics(train_metrics, val_metrics, output_dir):
    epochs = range(1, len(train_metrics) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Train')
    plt.plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [m['accuracy'] for m in train_metrics], 'b-', label='Train')
    plt.plot(epochs, [m['accuracy'] for m in val_metrics], 'r-', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [m['f1'] for m in train_metrics], 'b-', label='Train')
    plt.plot(epochs, [m['f1'] for m in val_metrics], 'r-', label='Validation')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def init_distributed_mode(config, args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config.distributed = False
        return

    config.distributed = True
    config.world_size = args.world_size
    config.rank = args.rank
    config.local_rank = args.local_rank
    
    torch.cuda.set_device(args.gpu)
    config.dist_backend = args.dist_backend
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Setup distributed training
    init_distributed_mode(config, args)
    
    # Setup environment
    setup_environment()
    
    # Update config with command line arguments
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.model_save_path = args.model_save_path
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    
    # Only print config on master process
    if not config.distributed or config.rank == 0:
        print('Training configuration:')
        for k, v in config.__dict__.items():
            print(f'  {k}: {v}')
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    # Save config
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=4)
    
    # Initialize model, criterion, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = ViTForProstateCancer(config).to(device)
    criterion = CombinedLoss(alpha=0.7)  # Adjust alpha to balance classification and segmentation loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_metric = float('inf')
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        start_epoch, checkpoint_metrics = load_checkpoint(model, optimizer, args.resume, device)
        best_metric = checkpoint_metrics.get('loss', float('inf'))
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Create a directory for saving checkpoints (only on master process)
    checkpoint_dir = os.path.join(config.output_dir, 'checkpoints')
    if not config.distributed or config.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    train_metrics = []
    val_metrics = []
    start_time = time.time()
    
    # Initialize progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, config.epochs), 
                     desc='Training Progress', 
                     position=0, 
                     leave=True,
                     bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
    
    for epoch in epoch_pbar:
        # Update progress bar
        epoch_pbar.set_description(f'Epoch {epoch+1}/{config.epochs}')
        
        # Train for one epoch
        train_metric = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config.epochs)
        train_metrics.append(train_metric)
        
        # Validate
        val_metric = validate_epoch(model, val_loader, criterion, device)
        val_metrics.append(val_metric)
        
        # Step the learning rate scheduler
        scheduler.step(val_metric['loss'])
        
        # Prepare metrics
        metrics = {
            'epoch': epoch + 1,
            'train': train_metric,
            'val': val_metric,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Print detailed metrics
        print('\n' + '='*80)
        print(f'Epoch {epoch+1}/{config.epochs} - Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        print('-'*80)
        print(f'Train Loss: {train_metric["loss"]:.4f} | ' \
              f'Cls Loss: {train_metric["cls_loss"]:.4f} | ' \
              f'Seg Loss: {train_metric["seg_loss"]:.4f} | ' \
              f'Acc: {train_metric["accuracy"]:.4f} | ' \
              f'F1: {train_metric["f1"]:.4f}')
        print(f'Val Loss: {val_metric["loss"]:.4f} | ' \
              f'Cls Loss: {val_metric["cls_loss"]:.4f} | ' \
              f'Seg Loss: {val_metric["seg_loss"]:.4f} | ' \
              f'Acc: {val_metric["accuracy"]:.4f} | ' \
              f'F1: {val_metric["f1"]:.4f}')
        print('='*80 + '\n')
        
        # Only save checkpoints on master process
        if not config.distributed or config.rank == 0:
            # Save checkpoint after each epoch with epoch number
            epoch_filename = f'model_epoch_{epoch+1:03d}.pth'
            epoch_path = os.path.join(checkpoint_dir, epoch_filename)
            save_checkpoint(
                model.module if hasattr(model, 'module') else model, 
                optimizer, 
                epoch, 
                {
                    'loss': val_metric['loss'],
                    'f1': val_metric['f1'],
                    'accuracy': val_metric['accuracy'],
                    'train_metrics': train_metric,
                    'val_metrics': val_metric
                }, 
                epoch_path
            )
            
            # Save best model if validation loss improved
            if val_metric['loss'] < best_metric:
                best_metric = val_metric['loss']
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    model.module if hasattr(model, 'module') else model, 
                    optimizer, 
                    epoch, 
                    {
                        'loss': best_metric,
                        'f1': val_metric['f1'],
                        'accuracy': val_metric['accuracy'],
                        'train_metrics': train_metric,
                        'val_metrics': val_metric
                    }, 
                    best_model_path
                )
                print(f'\n🔥 New best model saved! Val Loss: {best_metric:.4f} at Epoch {epoch+1}\n')
        
        # Only save metrics and plots on master process
        if not config.distributed or config.rank == 0:
            # Save metrics to file after each epoch
            metrics_file = os.path.join(config.output_dir, 'training_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump({
                    'train': train_metrics,
                    'val': val_metrics,
                    'best_metric': best_metric,
                    'best_epoch': epoch if val_metric['loss'] == best_metric else None
                }, f, indent=4)
            
            # Plot training metrics
            plot_training_metrics(train_metrics, val_metrics, config.output_dir)
            
            # Save a copy of the current metrics plot
            metrics_plot_path = os.path.join(config.output_dir, f'metrics_epoch_{epoch+1:03d}.png')
            plt.savefig(metrics_plot_path)
            plt.close()
        
        # Print remaining time estimation
        if epoch > 0:
            avg_time_per_epoch = (time.time() - start_time) / (epoch + 1 - start_epoch)
            remaining_epochs = config.epochs - (epoch + 1)
            remaining_time = avg_time_per_epoch * remaining_epochs
            print(f'\n⏳ Estimated time remaining: {remaining_time/60:.1f} minutes\n')
    
    print('Training complete!')

if __name__ == '__main__':
    main()
