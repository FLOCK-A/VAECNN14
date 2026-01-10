#!/usr/bin/env python3
"""
Sanity check script for the unified data interface.
Prints batch fields, shapes, and domain distribution.
"""
import sys
import json
import torch
import numpy as np
from collections import Counter

# Add parent directory to path
sys.path.insert(0, '/home/runner/work/VAECNN14/VAECNN14')

from data.dataloader import get_dataloader
from utils.data_validation import validate_no_leakage
from utils.domain_mask import create_domain_mask


def print_section(title):
    """Print a section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def sanity_check(dataset_json, data_root, label_key='scene', batch_size=8):
    """
    Run sanity checks on the dataset and dataloader.
    
    Args:
        dataset_json: Path to dataset JSON file
        data_root: Root directory for NPY features
        label_key: 'scene' or 'city'
        batch_size: Batch size for testing
    """
    print_section("UNIFIED DATA INTERFACE SANITY CHECK")
    
    print(f"\nConfiguration:")
    print(f"  Dataset JSON: {dataset_json}")
    print(f"  Data root: {data_root}")
    print(f"  Label key: {label_key}")
    print(f"  Batch size: {batch_size}")
    
    # Load dataset
    print_section("Loading Dataset")
    with open(dataset_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Splits found: {list(dataset.keys())}")
    for split_name, samples in dataset.items():
        print(f"  {split_name}: {len(samples)} samples")
    
    # Check first sample structure
    print_section("Sample Structure Check")
    first_split = list(dataset.keys())[0]
    if dataset[first_split]:
        sample = dataset[first_split][0]
        print(f"First sample in '{first_split}':")
        for key, value in sample.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
    
    # Data leakage validation
    print_section("Data Leakage Validation")
    try:
        # Example: validate training split only allows 'train' split
        train_samples = dataset.get('train', [])
        if train_samples:
            print("Validating train split (should only contain split='train')...")
            validate_no_leakage(train_samples, allowed_splits=['train'], phase='training')
            print("✓ Train split validation PASSED")
        
        # Example: validate unseen devices are not in training
        # Assuming domains 6,7,8 are unseen (s4,s5,s6)
        all_train_samples = dataset.get('train', []) + dataset.get('val', [])
        if all_train_samples:
            print("Validating no unseen devices in training (domains 6,7,8)...")
            validate_no_leakage(all_train_samples, 
                              allowed_devices=[0,1,2,3,4,5], 
                              phase='training')
            print("✓ Unseen device validation PASSED")
            
    except ValueError as e:
        print(f"✗ Validation FAILED:")
        print(str(e))
        return
    
    # Create dataloader
    print_section("Creating DataLoader")
    train_samples = dataset.get('train', [])
    if not train_samples:
        print("No train samples found, using first available split")
        first_split = list(dataset.keys())[0]
        train_samples = dataset[first_split]
    
    dataloader = get_dataloader(
        samples=train_samples,
        data_root=data_root,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        label_key=label_key
    )
    
    print(f"DataLoader created: {len(dataloader)} batches")
    
    # Get one batch
    print_section("Batch Fields and Shapes")
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    
    print(f"Batch fields:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:15s}: shape={list(value.shape):20s} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key:15s}: list with {len(value)} items, first={value[0] if value else 'N/A'}")
        else:
            print(f"  {key:15s}: type={type(value).__name__}")
    
    # Domain distribution
    print_section("Domain Distribution in Batch")
    domains = batch['domain'].numpy()
    domain_counts = Counter(domains)
    print(f"Domains in batch: {dict(domain_counts)}")
    
    if 'device_id' in batch:
        device_ids = batch['device_id']
        print(f"Device IDs: {set(device_ids)}")
    
    # Test domain mask
    print_section("Domain Mask Functionality")
    source_mask, target_mask = create_domain_mask(batch, source_domain=0)
    print(f"Source samples (domain=0): {source_mask.sum().item()}/{len(batch['domain'])}")
    print(f"Target samples (domain!=0): {target_mask.sum().item()}/{len(batch['domain'])}")
    print(f"Source mask: {source_mask.numpy()}")
    print(f"Target mask: {target_mask.numpy()}")
    
    # Check labels
    print_section("Label Information")
    print(f"Main label (label_key='{label_key}'):")
    print(f"  Values: {batch['label'].numpy()}")
    print(f"  Unique: {torch.unique(batch['label']).numpy()}")
    
    if 'scene_label' in batch:
        print(f"\nScene labels:")
        print(f"  Values: {batch['scene_label'].numpy()}")
        print(f"  Unique: {torch.unique(batch['scene_label']).numpy()}")
    
    if 'city_label' in batch:
        print(f"\nCity labels:")
        print(f"  Values: {batch['city_label'].numpy()}")
        print(f"  Unique: {torch.unique(batch['city_label']).numpy()}")
    
    # Feature shape
    print_section("Feature Information")
    features = batch['features']
    print(f"Features shape: {list(features.shape)}")
    print(f"Features dtype: {features.dtype}")
    print(f"Features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"Features mean: {features.mean().item():.4f}")
    print(f"Features std: {features.std().item():.4f}")
    
    # Split information
    if 'split' in batch:
        print_section("Split Information")
        splits = batch['split']
        split_counts = Counter(splits)
        print(f"Splits in batch: {dict(split_counts)}")
    
    print_section("Sanity Check COMPLETED Successfully!")
    print("\n✓ All checks passed!")
    print(f"✓ Batch contains {len(batch['features'])} samples")
    print(f"✓ Fields: {list(batch.keys())}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sanity check for unified data interface')
    parser.add_argument('--dataset_json', type=str, 
                        default='/home/runner/work/VAECNN14/VAECNN14/data/sample_dataset.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--data_root', type=str,
                        default='/home/runner/work/VAECNN14/VAECNN14/data',
                        help='Root directory for NPY features (can be dummy for testing)')
    parser.add_argument('--label_key', type=str, default='scene',
                        choices=['scene', 'city'],
                        help='Label key to use as main supervision')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    try:
        sanity_check(args.dataset_json, args.data_root, args.label_key, args.batch_size)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {type(e).__name__}")
        print(f"{'='*80}")
        print(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
