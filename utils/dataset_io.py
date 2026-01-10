#!/usr/bin/env python3
"""
Unified dataset JSON loading interface.

Supports multiple JSON formats and outputs a standardized structure for all scripts.
"""

import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def load_dataset(json_path: str) -> Dict[str, List[dict]]:
    """
    Load dataset from JSON file with flexible format support.
    
    Supports three input formats:
    1. Dict format: {"train": [...], "val": [...], "test": [...]}
    2. List with split field: [{"...", "split": "train"}, {"...", "split": "test"}, ...]
    3. List without split: [...] (first half=train, second half=test)
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Standardized dict with structure:
        {
            "train": List[dict],  # Training samples
            "val": List[dict],    # Validation samples (may be empty)
            "test": List[dict],   # Test samples
        }
        
        Each sample dict contains at least: file, domain, city, scene, split
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Case 1: Dict format with train/val/test keys
    if isinstance(data, dict):
        # Check if it has train/val/test structure
        if 'train' in data or 'val' in data or 'test' in data:
            result = {
                'train': data.get('train', []),
                'val': data.get('val', []),
                'test': data.get('test', []),
            }
            
            # Ensure split field is set for all samples
            for split_name, samples in result.items():
                for sample in samples:
                    if 'split' not in sample:
                        sample['split'] = split_name
            
            logger.info(f"Loaded dict-format dataset: train={len(result['train'])}, "
                       f"val={len(result['val'])}, test={len(result['test'])}")
            return result
        else:
            # Dict but not in expected format - treat as invalid
            raise ValueError(
                f"Dict-format JSON must have at least one of 'train', 'val', 'test' keys. "
                f"Found keys: {list(data.keys())}"
            )
    
    # Case 2 & 3: List format
    elif isinstance(data, list):
        if len(data) == 0:
            logger.warning(f"Empty dataset loaded from {json_path}")
            return {'train': [], 'val': [], 'test': []}
        
        # Check if samples have 'split' field
        has_split = all('split' in sample for sample in data)
        
        if has_split:
            # Case 2: List with split field
            result = {
                'train': [s for s in data if s.get('split') == 'train'],
                'val': [s for s in data if s.get('split') == 'val'],
                'test': [s for s in data if s.get('split') == 'test'],
            }
            
            logger.info(f"Loaded list-format dataset with split field: "
                       f"train={len(result['train'])}, val={len(result['val'])}, "
                       f"test={len(result['test'])}")
            return result
        else:
            # Case 3: List without split - split in half
            logger.warning(
                f"Dataset has no 'split' field. Splitting {len(data)} samples: "
                f"first half=train, second half=test"
            )
            
            split_point = len(data) // 2
            train_samples = data[:split_point]
            test_samples = data[split_point:]
            
            # Add split field to samples
            for sample in train_samples:
                sample['split'] = 'train'
            for sample in test_samples:
                sample['split'] = 'test'
            
            result = {
                'train': train_samples,
                'val': [],
                'test': test_samples,
            }
            
            logger.info(f"Split dataset into: train={len(result['train'])}, "
                       f"val={len(result['val'])}, test={len(result['test'])}")
            return result
    
    else:
        raise ValueError(
            f"Invalid JSON format. Expected dict or list, got {type(data).__name__}"
        )


def get_train_samples(json_path: str) -> List[dict]:
    """Convenience function to get only training samples."""
    dataset = load_dataset(json_path)
    return dataset['train']


def get_val_samples(json_path: str) -> List[dict]:
    """Convenience function to get only validation samples."""
    dataset = load_dataset(json_path)
    return dataset['val']


def get_test_samples(json_path: str) -> List[dict]:
    """Convenience function to get only test samples."""
    dataset = load_dataset(json_path)
    return dataset['test']


def get_train_val_samples(json_path: str) -> tuple:
    """Convenience function to get training and validation samples."""
    dataset = load_dataset(json_path)
    return dataset['train'], dataset['val']


def validate_sample_fields(sample: dict) -> bool:
    """
    Validate that a sample has required fields.
    
    Args:
        sample: Sample dict
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['file', 'domain', 'split']
    return all(field in sample for field in required_fields)
