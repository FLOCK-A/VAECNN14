#!/usr/bin/env python3
"""
Scene Teacher Training Script

Trains a scene classification teacher on:
- Source domain (A) training data only
- Supervision: scene labels

Output: outputs/teacher_scene/(global or A2{target})/best.pth + last.pth
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classifier import Cnn14Classifier
from data.dataloader import get_dataloader
from utils.data_validation import validate_no_leakage


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_validate_data(json_path, source_domain=0):
    """
    Load dataset and validate - only source domain
    
    Args:
        json_path: Path to dataset JSON
        source_domain: Source domain ID (default=0 for device A)
        
    Returns:
        source_samples (from split=train, domain=0 only)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Get training samples (split=train, domain=0 only)
    train_split = dataset.get('train', [])
    source_samples = [s for s in train_split if s.get('domain') == source_domain]
    
    # Validate no data leakage
    validate_no_leakage(source_samples,
                       allowed_splits=['train'],
                       allowed_devices=[source_domain],
                       phase='Scene Teacher training')
    
    print(f"Source domain (A) samples: {len(source_samples)}")
    
    return source_samples


def split_train_val(samples, val_ratio=0.1, seed=42):
    """Split samples into train and val sets"""
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    val_size = int(len(samples) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    return train_samples, val_samples


def count_parameters(model, trainable_only=True):
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['scene_label'].to(device)  # Use scene labels
        
        optimizer.zero_grad()
        
        # Forward pass (returns raw logits)
        logits = model(features)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * features.size(0)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['scene_label'].to(device)
            
            # Forward pass
            logits = model(features)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Statistics
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += features.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Scene Teacher')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to dataset JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for NPY features')
    parser.add_argument('--target_name', type=str, default='global',
                       help='Target device name or "global" for general model')
    parser.add_argument('--num_scenes', type=int, default=10,
                       help='Number of scene classes')
    parser.add_argument('--num_cities', type=int, default=12,
                       help='Number of city classes (not used, for API consistency)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/teacher_scene',
                       help='Output directory for checkpoints')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio from training data')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and validate data (source domain only)
    print("\n" + "="*80)
    print("Loading and validating data (source domain only)...")
    print("="*80)
    source_samples = load_and_validate_data(args.json_path)
    
    # Split train/val
    train_samples, val_samples = split_train_val(source_samples, args.val_ratio, args.seed)
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create dataloaders (use scene labels)
    train_loader = get_dataloader(
        train_samples,
        data_root=args.data_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        label_key='scene'  # Use scene labels
    )
    
    val_loader = get_dataloader(
        val_samples,
        data_root=args.data_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        label_key='scene'
    )
    
    # Create model
    print("\n" + "="*80)
    print("Creating Scene Teacher model...")
    print("="*80)
    model = Cnn14Classifier(
        classes_num=args.num_scenes,
        checkpoint_path=None,  # No pretrained checkpoint for now
        use_projection_head=False,
        dropout=0.3,
        use_adapt_head=False
    )
    model = model.to(device)
    
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_acc = 0.0
    
    # Output directory: global or A2{target}
    if args.target_name == 'global':
        output_subdir = os.path.join(args.output_dir, 'global')
    else:
        output_subdir = os.path.join(args.output_dir, f'A2{args.target_name}')
    
    os.makedirs(output_subdir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(output_subdir, 'best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save last model
    last_path = os.path.join(output_subdir, 'last.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, last_path)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {output_subdir}/")
    print(f"  - best.pth (val_acc: {best_val_acc:.2f}%)")
    print(f"  - last.pth (val_acc: {val_acc:.2f}%)")


if __name__ == '__main__':
    main()
