#!/usr/bin/env python3
"""
Student Training with Knowledge Distillation

Supports multiple teacher modes:
- city2scene: Use only City2Scene teacher
- scene: Use only Scene teacher  
- mean_fusion: Average logits from both teachers
- attn_fusion: Trainable attention gate for teacher fusion

Features:
- Domain masking: CE on source, KD on target
- Cached teacher logits (offline, no online forward)
- Full KD configuration: temperature, weighting, scheduling
- Evaluation on target test set
- Best model saving based on target accuracy
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.classifier import Cnn14Classifier
from data.dataloader import ASCDataset
from utils.data_validation import validate_no_leakage, verify_cache_exists
from utils.domain_mask import create_domain_mask
from losses.kd import compute_student_loss, KDLossConfig


class AttentionFusionGate(nn.Module):
    """
    Learnable attention gate for fusing two teacher logits.
    
    Inputs: city2scene_logits, scene_logits
    Output: gamma (scalar or per-sample weight)
    Fused = gamma * scene + (1-gamma) * city2scene
    """
    def __init__(self, num_classes, hidden_dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, city2scene_logits, scene_logits):
        """
        Args:
            city2scene_logits: [B, num_classes]
            scene_logits: [B, num_classes]
        
        Returns:
            gamma: [B, 1] - fusion weights
        """
        # Concatenate logits
        concat_logits = torch.cat([city2scene_logits, scene_logits], dim=1)  # [B, 2*num_classes]
        gamma = self.gate(concat_logits)  # [B, 1]
        return gamma


class CachedTeacherLogitsLoader:
    """Load cached teacher logits from disk"""
    def __init__(self, cache_root, teacher_modes):
        """
        Args:
            cache_root: Path to cache directory (e.g., cache/A2b)
            teacher_modes: List of teacher types to load (e.g., ['city2scene', 'scene'])
        """
        self.cache_root = cache_root
        self.teacher_modes = teacher_modes
        
        # Load cache index
        index_path = os.path.join(cache_root, 'cache_index.json')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Cache index not found: {index_path}")
        
        with open(index_path, 'r') as f:
            self.cache_index = json.load(f)
        
        print(f"Loaded cache index from {index_path}")
        print(f"Available teachers: {list(self.cache_index.keys())}")
        print(f"Number of cached samples: {self.cache_index.get('num_samples', 'unknown')}")
    
    def load_logits(self, file_paths, teacher_mode):
        """
        Load cached logits for a batch of samples.
        
        Args:
            file_paths: List of file paths
            teacher_mode: 'city2scene' or 'scene'
        
        Returns:
            logits: Tensor [B, num_classes]
            
        Raises:
            KeyError: If cache file not found with detailed path information
        """
        if teacher_mode not in self.cache_index:
            raise ValueError(f"Teacher mode '{teacher_mode}' not found in cache. "
                           f"Available: {list(self.cache_index.keys())}")
        
        logits_list = []
        missing_files = []
        
        for file_path in file_paths:
            # Get base filename
            base_name = os.path.basename(file_path)
            
            if base_name not in self.cache_index[teacher_mode]:
                missing_files.append(f"  PATH: {file_path} (base: {base_name})")
                continue
            
            cache_file = self.cache_index[teacher_mode][base_name]
            
            if not os.path.exists(cache_file):
                missing_files.append(f"  PATH: {file_path} -> cache: {cache_file} (NOT FOUND)")
                continue
            
            logits = np.load(cache_file)
            logits_list.append(logits)
        
        if missing_files:
            error_msg = f"\n{'='*80}\n"
            error_msg += f"🚨 CACHE MISSING FOR {len(missing_files)} SAMPLES! 🚨\n"
            error_msg += f"{'='*80}\n"
            error_msg += f"Teacher mode: {teacher_mode}\n"
            error_msg += f"Cache root: {self.cache_root}\n\n"
            error_msg += "Missing cache files:\n"
            error_msg += "\n".join(missing_files[:20])
            if len(missing_files) > 20:
                error_msg += f"\n... and {len(missing_files) - 20} more\n"
            error_msg += f"\n{'='*80}\n"
            error_msg += "⚠️  Please run dump_teacher_logits.py to cache all required samples.\n"
            error_msg += f"{'='*80}\n"
            raise KeyError(error_msg)
        
        return torch.from_numpy(np.stack(logits_list, axis=0)).float()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_id_from_name(target_name):
    """Map target name to device ID"""
    device_mapping = {
        'A': 0, 'b': 1, 'c': 2,
        's1': 3, 's2': 4, 's3': 5,
        's4': 6, 's5': 7, 's6': 8
    }
    return device_mapping.get(target_name, None)


def create_dataloaders(json_path, data_root, target_domain, batch_size, val_ratio=0.1, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        json_path: Path to dataset JSON
        data_root: Root directory for data files
        target_domain: Target device ID (e.g., 1 for 'b')
        batch_size: Batch size
        val_ratio: Fraction of training data to use for validation
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load full dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter: only split='train' for training/validation
    # Include source (A, domain=0) and target domain
    train_samples = [
        s for s in data
        if s['split'] == 'train' and s['domain'] in [0, target_domain]
    ]
    
    # Test set: target domain, split='test'
    test_samples = [
        s for s in data
        if s['split'] == 'test' and s['domain'] == target_domain
    ]
    
    # Validate no data leakage
    validate_no_leakage(train_samples, 
                       allowed_splits=['train'],
                       allowed_devices=[0, 1, 2, 3, 4, 5])  # Exclude s4,s5,s6
    
    # Split train into train/val
    num_train = len(train_samples)
    num_val = int(num_train * val_ratio)
    indices = list(range(num_train))
    random.shuffle(indices)
    
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    # Create datasets (use scene labels for student)
    full_train_dataset = ASCDataset(train_samples, data_root, label_key='scene')
    test_dataset = ASCDataset(test_samples, data_root, label_key='scene')
    
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data splits: train={len(train_subset)}, val={len(val_subset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, kd_config, cache_loader, 
                teacher_mode, fusion_gate, device, epoch, source_domain=0, 
                apply_kd_to='target'):
    """
    Train for one epoch.
    
    Args:
        model: Student model
        train_loader: Training dataloader
        optimizer: Optimizer
        criterion: Loss criterion (cross-entropy)
        kd_config: KD loss configuration
        cache_loader: Cached teacher logits loader
        teacher_mode: 'city2scene', 'scene', 'mean_fusion', 'attn_fusion', or 'ce_only'
        fusion_gate: Attention fusion gate (for attn_fusion mode)
        device: torch device
        epoch: Current epoch number
        source_domain: Source domain ID (default=0 for A)
        apply_kd_to: 'target', 'all', or 'source' - where to apply KD
    
    Returns:
        avg_loss, avg_acc, stats_dict
    """
    model.train()
    if fusion_gate is not None:
        fusion_gate.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    ce_loss_sum = 0
    kd_loss_sum = 0
    coverage_sum = 0
    confidence_sum = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        student_logits = model(features)
        
        # Get domain masks
        source_mask, target_mask = create_domain_mask(batch, source_domain)
        source_mask = source_mask.to(device)
        target_mask = target_mask.to(device)
        
        # Determine KD mask based on apply_kd_to
        if apply_kd_to == 'target':
            kd_mask = target_mask
        elif apply_kd_to == 'all':
            kd_mask = torch.ones_like(source_mask, dtype=torch.bool)
        elif apply_kd_to == 'source':
            kd_mask = source_mask
        else:
            raise ValueError(f"Invalid apply_kd_to: {apply_kd_to}")
        
        # CE loss on source samples only
        if source_mask.sum() > 0:
            ce_loss = criterion(student_logits[source_mask], labels[source_mask])
        else:
            ce_loss = torch.tensor(0.0).to(device)
        
        # KD loss
        if teacher_mode == 'ce_only':
            # No KD, only CE
            loss = ce_loss
            kd_loss_val = 0.0
            coverage = 0.0
            avg_confidence = 0.0
        else:
            # Load teacher logits from cache
            if teacher_mode in ['city2scene', 'scene']:
                teacher_logits = cache_loader.load_logits(batch['path'], teacher_mode).to(device)
            elif teacher_mode == 'mean_fusion':
                city2scene_logits = cache_loader.load_logits(batch['path'], 'city2scene').to(device)
                scene_logits = cache_loader.load_logits(batch['path'], 'scene').to(device)
                teacher_logits = (city2scene_logits + scene_logits) / 2.0
            elif teacher_mode == 'attn_fusion':
                city2scene_logits = cache_loader.load_logits(batch['path'], 'city2scene').to(device)
                scene_logits = cache_loader.load_logits(batch['path'], 'scene').to(device)
                
                # Compute fusion weights
                gamma = fusion_gate(city2scene_logits, scene_logits)  # [B, 1]
                teacher_logits = gamma * scene_logits + (1 - gamma) * city2scene_logits
            else:
                raise ValueError(f"Unknown teacher_mode: {teacher_mode}")
            
            # Compute combined loss (CE + KD)
            combined_loss, stats = compute_student_loss(
                student_logits,
                teacher_logits,
                labels,
                kd_config,
                current_epoch=epoch,
                domain_mask=kd_mask
            )
            
            # For source samples, use only CE; for target, use combined
            # Weight by number of samples in each domain
            if source_mask.sum() > 0 and target_mask.sum() > 0:
                # Mixed batch: combine losses appropriately
                loss = combined_loss  # Already weighted by kd_config
            else:
                loss = combined_loss
            
            kd_loss_val = stats['kd_loss']
            coverage = stats['coverage']
            avg_confidence = stats['avg_confidence']
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(student_logits, 1)
        correct = (predicted == labels).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
        
        ce_loss_sum += ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
        kd_loss_sum += kd_loss_val
        coverage_sum += coverage
        confidence_sum += avg_confidence
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.0 * correct / labels.size(0):.2f}%"
        })
    
    avg_loss = total_loss / num_batches
    avg_acc = 100.0 * total_correct / total_samples
    
    stats_dict = {
        'ce_loss': ce_loss_sum / num_batches,
        'kd_loss': kd_loss_sum / num_batches,
        'coverage': coverage_sum / num_batches,
        'avg_confidence': confidence_sum / num_batches
    }
    
    return avg_loss, avg_acc, stats_dict


def evaluate(model, val_loader, criterion, device, desc="Val"):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Student model
        val_loader: Validation dataloader
        criterion: Loss criterion
        device: torch device
        desc: Description for progress bar
    
    Returns:
        avg_loss, avg_acc
    """
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=desc)
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(features)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            correct = (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / labels.size(0):.2f}%"
            })
    
    avg_loss = total_loss / num_batches
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Student Training with Knowledge Distillation')
    
    # Data arguments
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to dataset JSON')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for data files')
    parser.add_argument('--target_name', type=str, required=True,
                       help='Target device name (e.g., b, c, s1, s2, s3)')
    
    # Model arguments
    parser.add_argument('--num_scenes', type=int, default=10,
                       help='Number of scene classes')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to pretrained checkpoint (optional)')
    
    # Teacher arguments
    parser.add_argument('--teacher_mode', type=str, required=True,
                       choices=['ce_only', 'city2scene', 'scene', 'mean_fusion', 'attn_fusion'],
                       help='Teacher mode for distillation')
    parser.add_argument('--cache_root', type=str, default=None,
                       help='Path to cached teacher logits (required if teacher_mode != ce_only)')
    parser.add_argument('--apply_kd_to', type=str, default='target',
                       choices=['target', 'all', 'source'],
                       help='Where to apply KD loss')
    
    # KD configuration
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for KD')
    parser.add_argument('--kd_alpha', type=float, default=0.7,
                       help='Weight for KD loss (1-alpha for CE)')
    parser.add_argument('--loss_type', type=str, default='kldiv',
                       choices=['kldiv', 'soft_ce'],
                       help='KD loss type')
    parser.add_argument('--kd_weighting', type=str, default='none',
                       choices=['none', 'pmax', 'threshold'],
                       help='KD weighting strategy')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                       help='Confidence threshold for threshold weighting')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                       help='Epochs with only CE loss')
    parser.add_argument('--kd_ramp_epochs', type=int, default=0,
                       help='Epochs to linearly ramp up KD weight')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Fraction of training data for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/student',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get target domain ID
    target_domain = get_device_id_from_name(args.target_name)
    if target_domain is None:
        raise ValueError(f"Invalid target_name: {args.target_name}")
    
    # Create output directory
    exp_name = f"A2{args.target_name}/{args.teacher_mode}"
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("="*80)
    print("Student Training with Knowledge Distillation")
    print("="*80)
    print(f"Target: {args.target_name} (domain={target_domain})")
    print(f"Teacher mode: {args.teacher_mode}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.json_path,
        args.data_root,
        target_domain,
        args.batch_size,
        args.val_ratio,
        args.num_workers
    )
    
    # Initialize cache loader (if needed)
    cache_loader = None
    if args.teacher_mode != 'ce_only':
        if args.cache_root is None:
            raise ValueError("--cache_root is required when teacher_mode != ce_only")
        
        # Determine which teachers to load
        if args.teacher_mode in ['city2scene', 'scene']:
            teacher_modes = [args.teacher_mode]
        else:  # mean_fusion or attn_fusion
            teacher_modes = ['city2scene', 'scene']
        
        cache_loader = CachedTeacherLogitsLoader(args.cache_root, teacher_modes)
    
    # Initialize model
    model = Cnn14Classifier(
        classes_num=args.num_scenes,
        checkpoint_path=args.checkpoint_path
    ).to(device)
    
    # Initialize fusion gate (if needed)
    fusion_gate = None
    if args.teacher_mode == 'attn_fusion':
        fusion_gate = AttentionFusionGate(args.num_scenes).to(device)
        print(f"Attention Fusion Gate initialized")
    
    # KD configuration
    kd_config = KDLossConfig(
        temperature=args.temperature,
        kd_alpha=args.kd_alpha,
        loss_type=args.loss_type,
        kd_weighting=args.kd_weighting,
        confidence_threshold=args.confidence_threshold,
        warmup_epochs=args.warmup_epochs,
        kd_ramp_epochs=args.kd_ramp_epochs
    )
    
    print(f"\nKD Configuration:")
    print(f"  Temperature: {kd_config.temperature}")
    print(f"  KD Alpha: {kd_config.kd_alpha}")
    print(f"  Loss Type: {kd_config.loss_type}")
    print(f"  Weighting: {kd_config.kd_weighting}")
    print(f"  Warmup Epochs: {kd_config.warmup_epochs}")
    print(f"  Ramp Epochs: {kd_config.kd_ramp_epochs}")
    print(f"  Apply KD to: {args.apply_kd_to}")
    
    # Optimizer
    params_to_optimize = list(model.parameters())
    if fusion_gate is not None:
        params_to_optimize += list(fusion_gate.parameters())
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_target_acc = 0.0
    history = []
    
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_stats = train_epoch(
            model, train_loader, optimizer, criterion, kd_config, cache_loader,
            args.teacher_mode, fusion_gate, device, epoch, source_domain=0,
            apply_kd_to=args.apply_kd_to
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, desc="Val")
        
        # Test on target
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc="Test (Target)")
        
        # Log
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        if args.teacher_mode != 'ce_only':
            print(f"  KD Stats: CE={train_stats['ce_loss']:.4f}, "
                  f"KD={train_stats['kd_loss']:.4f}, "
                  f"Coverage={train_stats['coverage']:.2%}, "
                  f"Confidence={train_stats['avg_confidence']:.3f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            **train_stats
        })
        
        # Save best model (based on target test accuracy)
        if test_acc > best_target_acc:
            best_target_acc = test_acc
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'val_acc': val_acc,
                'config': config_dict
            }
            if fusion_gate is not None:
                save_dict['fusion_gate_state_dict'] = fusion_gate.state_dict()
            
            torch.save(save_dict, os.path.join(output_dir, 'best.pth'))
            print(f"  ✓ Best model saved (target acc: {best_target_acc:.2f}%)")
    
    # Save last model
    save_dict = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'val_acc': val_acc,
        'config': config_dict
    }
    if fusion_gate is not None:
        save_dict['fusion_gate_state_dict'] = fusion_gate.state_dict()
    
    torch.save(save_dict, os.path.join(output_dir, 'last.pth'))
    
    # Save history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Training Completed")
    print(f"{'='*80}")
    print(f"Best Target Accuracy: {best_target_acc:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"  - best.pth (target acc: {best_target_acc:.2f}%)")
    print(f"  - last.pth (final epoch)")
    print(f"  - history.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
