import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from tqdm import tqdm
import random

from models import PolarDETR
from data.dataset import DentalDataModule
from utils.metrics import calculate_metrics
from utils.visualization import visualize_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='PolarDETR Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (smaller dataset, more logs)')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transforms(config):
    """Get image transforms for each split"""
    img_size = tuple(config['data']['image_size'])
    
    train_transform = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': test_transform,
        'test': test_transform
    }

def save_checkpoint(model, optimizer, scheduler, epoch, best_score, config, output_dir, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_score': best_score,
        'config': config
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def train_epoch(model, dataloader, optimizer, device, epoch, config, scaler=None, writer=None):
    """Train model for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_detr_loss = 0.0
    total_anatomy_loss = 0.0
    total_position_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for i, (images, texts, targets) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if available
        if scaler:
            with autocast():
                outputs = model(images, texts, targets)
                loss = outputs['losses']['total_loss']
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config['training']['clip_max_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_max_norm'])
            
            # Update weights with scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular forward and backward pass
            outputs = model(images, texts, targets)
            loss = outputs['losses']['total_loss']
            loss.backward()
            
            # Gradient clipping
            if config['training']['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_max_norm'])
            
            # Update weights
            optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_detr_loss += outputs['losses']['detr_loss'].item()
        total_anatomy_loss += outputs['losses']['anatomy_loss'].item()
        total_position_loss += outputs['losses']['position_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'detr': outputs['losses']['detr_loss'].item(),
            'anatomy': outputs['losses']['anatomy_loss'].item(),
            'position': outputs['losses']['position_loss'].item()
        })
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_detr_loss = total_detr_loss / len(dataloader)
    avg_anatomy_loss = total_anatomy_loss / len(dataloader)
    avg_position_loss = total_position_loss / len(dataloader)
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Train/DETR_Loss', avg_detr_loss, epoch)
        writer.add_scalar('Train/Anatomy_Loss', avg_anatomy_loss, epoch)
        writer.add_scalar('Train/Position_Loss', avg_position_loss, epoch)
    
    return avg_loss, avg_detr_loss, avg_anatomy_loss, avg_position_loss

def validate(model, dataloader, device, epoch, config, writer=None):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_detr_loss = 0.0
    total_anatomy_loss = 0.0
    total_position_loss = 0.0
    
    all_preds = []
    all_targets = []
    all_texts = []
    all_images = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, (images, texts, targets) in enumerate(progress_bar):
            # Move data to device
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Forward pass
            outputs = model(images, texts, targets)
            loss = outputs['losses']['total_loss']
            
            # Track losses
            total_loss += loss.item()
            total_detr_loss += outputs['losses']['detr_loss'].item()
            total_anatomy_loss += outputs['losses']['anatomy_loss'].item()
            total_position_loss += outputs['losses']['position_loss'].item()
            
            # Track predictions and targets for metrics
            all_preds.append(outputs['pred_boxes'])
            all_targets.extend(targets)
            all_texts.extend(texts)
            
            # Save some images for visualization
            if i < 5:
                all_images.append(images)
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_detr_loss = total_detr_loss / len(dataloader)
    avg_anatomy_loss = total_anatomy_loss / len(dataloader)
    avg_position_loss = total_position_loss / len(dataloader)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets, config)
    
    # Visualize some predictions
    if writer and all_images:
        vis_images = torch.cat(all_images[:2], dim=0)
        vis_preds = torch.cat(all_preds[:2], dim=0)
        vis_texts = all_texts[:2 * vis_images.size(0)]
        vis_targets = all_targets[:2 * vis_images.size(0)]
        
        vis_fig = visualize_predictions(
            vis_images, vis_preds, vis_texts, vis_targets, config
        )
        if vis_fig:
            writer.add_figure('Validation/Predictions', vis_fig, epoch)
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Val/DETR_Loss', avg_detr_loss, epoch)
        writer.add_scalar('Val/Anatomy_Loss', avg_anatomy_loss, epoch)
        writer.add_scalar('Val/Position_Loss', avg_position_loss, epoch)
        
        # Log metrics
        for k, v in metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
    
    return avg_loss, metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transforms = get_transforms(config)
    
    # Create data module
    data_module = DentalDataModule(
        data_dir=args.data_dir,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        transforms=transforms
    )
    
    # Create dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    # Create model
    model = PolarDETR(config)
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_drop'],
        gamma=0.1
    )
    
    # Enable automatic mixed precision if available
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_score = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict') and scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_score = checkpoint.get('best_score', 0.0)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    num_epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, detr_loss, anatomy_loss, position_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler, writer
        )
        
        # Validate
        val_loss, metrics = validate(model, val_loader, device, epoch, config, writer)
        
        # Update scheduler
        scheduler.step()
        
        # Update best score (using mAP as primary metric)
        current_score = metrics.get('mAP', 0.0)
        is_best = current_score > best_score
        
        if is_best:
            best_score = current_score
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score,
                config, output_dir, 'best_model.pth'
            )
        
        # Save regular checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_score,
            config, output_dir, 'last_model.pth'
        )
        
        # Print epoch summary
        print(f"Epoch {epoch}/{num_epochs-1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Metrics: {metrics}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.8f}")
        print(f"  Best Score: {best_score:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, num_epochs-1, best_score,
        config, output_dir, 'final_model.pth'
    )
    
    # Close tensorboard writer
    writer.close()
    
    # Final message
    print(f"Training completed. Best score: {best_score:.4f}")
    print(f"Models saved to {output_dir}")

if __name__ == '__main__':
    main() 