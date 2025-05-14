#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
import yaml

class DentalEntityDataset(Dataset):
    """Dataset for dental entity extraction"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = item['text']
        
        # Encode text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Get labels
        tooth_number = item.get('tooth_number', 0)
        distance = item.get('distance', 0.0)
        direction = item.get('direction', 0)
        quadrant = item.get('quadrant', 0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tooth_number': torch.tensor(tooth_number, dtype=torch.float),
            'distance': torch.tensor(distance, dtype=torch.float),
            'direction': torch.tensor(direction, dtype=torch.long),
            'quadrant': torch.tensor(quadrant, dtype=torch.long)
        }

class DentalEntityExtractor(nn.Module):
    """Dental entity extraction model based on BioClinicalBERT"""
    
    def __init__(self, pretrained_model_name, freeze_bert=False):
        super().__init__()
        
        # Load BioClinicalBERT
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT if needed
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Task-specific heads
        self.tooth_number_head = nn.Linear(self.hidden_size, 1)
        self.distance_head = nn.Linear(self.hidden_size, 1)
        self.direction_classifier = nn.Linear(self.hidden_size, 8)  # 8 possible directions
        self.quadrant_classifier = nn.Linear(self.hidden_size, 4)  # 4 quadrants
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Get predictions
        tooth_number = torch.sigmoid(self.tooth_number_head(cls_output)) * 48  # Scale to 1-48 range
        distance = torch.relu(self.distance_head(cls_output))  # Distance can't be negative
        direction_logits = self.direction_classifier(cls_output)
        quadrant_logits = self.quadrant_classifier(cls_output)
        
        return tooth_number, distance, direction_logits, quadrant_logits

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train model for one epoch"""
    model.train()
    
    epoch_loss = 0
    tooth_mse = 0
    distance_mse = 0
    direction_acc = 0
    quadrant_acc = 0
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    for batch in tqdm(data_loader, desc="Training"):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tooth_number = batch['tooth_number'].to(device)
        distance = batch['distance'].to(device)
        direction = batch['direction'].to(device)
        quadrant = batch['quadrant'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        tooth_pred, distance_pred, direction_logits, quadrant_logits = model(input_ids, attention_mask)
        
        # Calculate losses
        tooth_loss = mse_loss(tooth_pred.squeeze(), tooth_number)
        distance_loss = mse_loss(distance_pred.squeeze(), distance)
        direction_loss = ce_loss(direction_logits, direction)
        quadrant_loss = ce_loss(quadrant_logits, quadrant)
        
        # Combined loss
        loss = tooth_loss + distance_loss + direction_loss + quadrant_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update statistics
        epoch_loss += loss.item()
        tooth_mse += tooth_loss.item()
        distance_mse += distance_loss.item()
        
        # Calculate accuracies
        direction_pred = torch.argmax(direction_logits, dim=1)
        quadrant_pred = torch.argmax(quadrant_logits, dim=1)
        
        direction_acc += (direction_pred == direction).sum().item() / len(direction)
        quadrant_acc += (quadrant_pred == quadrant).sum().item() / len(quadrant)
    
    # Calculate average metrics
    num_batches = len(data_loader)
    return {
        'loss': epoch_loss / num_batches,
        'tooth_mse': tooth_mse / num_batches,
        'distance_mse': distance_mse / num_batches,
        'direction_acc': direction_acc / num_batches,
        'quadrant_acc': quadrant_acc / num_batches
    }

def validate(model, data_loader, device):
    """Validate model"""
    model.eval()
    
    val_loss = 0
    tooth_mse = 0
    distance_mse = 0
    direction_acc = 0
    quadrant_acc = 0
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Collect predictions and targets for F1 score
    direction_preds = []
    direction_targets = []
    quadrant_preds = []
    quadrant_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tooth_number = batch['tooth_number'].to(device)
            distance = batch['distance'].to(device)
            direction = batch['direction'].to(device)
            quadrant = batch['quadrant'].to(device)
            
            # Forward pass
            tooth_pred, distance_pred, direction_logits, quadrant_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            tooth_loss = mse_loss(tooth_pred.squeeze(), tooth_number)
            distance_loss = mse_loss(distance_pred.squeeze(), distance)
            direction_loss = ce_loss(direction_logits, direction)
            quadrant_loss = ce_loss(quadrant_logits, quadrant)
            
            # Combined loss
            loss = tooth_loss + distance_loss + direction_loss + quadrant_loss
            
            # Update statistics
            val_loss += loss.item()
            tooth_mse += tooth_loss.item()
            distance_mse += distance_loss.item()
            
            # Calculate accuracies and collect predictions
            direction_pred = torch.argmax(direction_logits, dim=1)
            quadrant_pred = torch.argmax(quadrant_logits, dim=1)
            
            direction_acc += (direction_pred == direction).sum().item() / len(direction)
            quadrant_acc += (quadrant_pred == quadrant).sum().item() / len(quadrant)
            
            # Collect predictions and targets for F1 score
            direction_preds.extend(direction_pred.cpu().numpy())
            direction_targets.extend(direction.cpu().numpy())
            quadrant_preds.extend(quadrant_pred.cpu().numpy())
            quadrant_targets.extend(quadrant.cpu().numpy())
    
    # Calculate F1 scores
    direction_f1 = f1_score(direction_targets, direction_preds, average='macro')
    quadrant_f1 = f1_score(quadrant_targets, quadrant_preds, average='macro')
    
    # Calculate average metrics
    num_batches = len(data_loader)
    return {
        'loss': val_loss / num_batches,
        'tooth_mse': tooth_mse / num_batches,
        'distance_mse': distance_mse / num_batches,
        'direction_acc': direction_acc / num_batches,
        'quadrant_acc': quadrant_acc / num_batches,
        'direction_f1': direction_f1,
        'quadrant_f1': quadrant_f1
    }

def save_model(model, tokenizer, config, save_path):
    """Save model and configuration"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.bert.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save task heads
    torch.save({
        'tooth_number_head': model.tooth_number_head.state_dict(),
        'distance_head': model.distance_head.state_dict(),
        'direction_classifier': model.direction_classifier.state_dict(),
        'quadrant_classifier': model.quadrant_classifier.state_dict()
    }, os.path.join(save_path, 'task_heads.pt'))
    
    # Save configuration
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Model saved to {save_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune BioClinicalBERT for dental entity extraction")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained'])
    
    # Create datasets
    train_dataset = DentalEntityDataset(
        config['data']['train_path'],
        tokenizer,
        max_length=config['data']['max_length']
    )
    
    # Split into train and validation
    if config['data'].get('val_path'):
        val_dataset = DentalEntityDataset(
            config['data']['val_path'],
            tokenizer,
            max_length=config['data']['max_length']
        )
    else:
        # Split train dataset
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Create model
    model = DentalEntityExtractor(
        config['model']['pretrained'],
        freeze_bert=config['model'].get('freeze_bert', False)
    )
    model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.1))
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train epoch
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Tooth MSE: {train_metrics['tooth_mse']:.4f}, "
              f"Distance MSE: {train_metrics['distance_mse']:.4f}, "
              f"Direction Acc: {train_metrics['direction_acc']:.4f}, "
              f"Quadrant Acc: {train_metrics['quadrant_acc']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Tooth MSE: {val_metrics['tooth_mse']:.4f}, "
              f"Distance MSE: {val_metrics['distance_mse']:.4f}, "
              f"Direction Acc: {val_metrics['direction_acc']:.4f} (F1: {val_metrics['direction_f1']:.4f}), "
              f"Quadrant Acc: {val_metrics['quadrant_acc']:.4f} (F1: {val_metrics['quadrant_f1']:.4f})")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            print(f"New best model (loss: {best_val_loss:.4f})")
            save_model(model, tokenizer, config, config['model']['save_path'])
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main() 