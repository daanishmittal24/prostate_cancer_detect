import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        if len(batch) == 3:  # image, mask, label
            images, masks, labels = batch
            masks = masks.to(device)
        else:  # image, label
            images, labels = batch
            masks = None
            
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, pred_masks = model(images)
        
        # Calculate loss
        loss, cls_loss, seg_loss = criterion(logits, pred_masks, labels, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        running_cls_loss += cls_loss.item()
        if seg_loss != 0:
            running_seg_loss += seg_loss.item()
        
        # Get predictions
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'cls_loss': running_cls_loss / (batch_idx + 1),
            'seg_loss': running_seg_loss / (batch_idx + 1) if seg_loss != 0 else 0
        })
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_seg_loss = running_seg_loss / len(dataloader) if running_seg_loss > 0 else 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    metrics = {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'seg_loss': avg_seg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa
    }
    
    return metrics

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            if len(batch) == 3:  # image, mask, label
                images, masks, labels = batch
                masks = masks.to(device)
            else:  # image, label
                images, labels = batch
                masks = None
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, pred_masks = model(images)
            
            # Calculate loss
            loss, cls_loss, seg_loss = criterion(logits, pred_masks, labels, masks)
            
            # Update statistics
            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            if seg_loss != 0:
                running_seg_loss += seg_loss.item()
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_seg_loss = running_seg_loss / len(dataloader) if running_seg_loss > 0 else 0
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    metrics = {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'seg_loss': avg_seg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'kappa': kappa
    }
    
    return metrics

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filename)

def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']
