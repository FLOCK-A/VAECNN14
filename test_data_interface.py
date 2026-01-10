#!/usr/bin/env python3
"""
Simple test script that validates the dataloader modifications work correctly
without requiring actual NPY files.
"""
import sys
import json
import os

sys.path.insert(0, '/home/runner/work/VAECNN14/VAECNN14')

# Mock numpy before importing dataloader
class MockArray:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = 'float32'
    
    def __repr__(self):
        return f"MockArray(shape={self.shape})"

import numpy as np
original_load = np.load

def mock_load(path, *args, **kwargs):
    """Mock np.load to return dummy arrays"""
    if path.endswith('.npy'):
        # Return a dummy array shape (100 time steps, 64 mel bins)
        return np.random.randn(100, 64).astype(np.float32)
    return original_load(path, *args, **kwargs)

# Patch numpy.load
np.load = mock_load

from data.dataloader import ASCDataset, get_dataloader
from utils.data_validation import validate_no_leakage
from utils.domain_mask import create_domain_mask


def test_dataset_fields():
    """Test that dataset returns all required fields"""
    print("="*80)
    print("TEST: Dataset Field Validation")
    print("="*80)
    
    # Create a sample
    sample = {
        'file': 'test.npy',
        'scene_label': 0,
        'city_label': 1,
        'domain': 0,
        'split': 'train',
        'device_id': 'A'
    }
    
    dataset = ASCDataset([sample], data_root='/tmp', label_key='scene')
    item = dataset[0]
    
    required_fields = ['features', 'scene_label', 'city_label', 'domain', 
                       'split', 'device_id', 'path', 'label']
    
    print(f"\nChecking required fields in dataset output:")
    for field in required_fields:
        if field in item:
            print(f"  ✓ {field}: {type(item[field])}")
        else:
            print(f"  ✗ {field}: MISSING")
            return False
    
    # Check backward compatibility
    if item['label'].item() == item['scene_label'].item():
        print(f"\n✓ Backward compatibility: 'label' matches 'scene_label' when label_key='scene'")
    else:
        print(f"\n✗ Backward compatibility failed")
        return False
    
    # Test city label_key
    dataset_city = ASCDataset([sample], data_root='/tmp', label_key='city')
    item_city = dataset_city[0]
    if item_city['label'].item() == item_city['city_label'].item():
        print(f"✓ Label key switching: 'label' matches 'city_label' when label_key='city'")
    else:
        print(f"✗ Label key switching failed")
        return False
    
    print("\n✓ TEST PASSED: All required fields present")
    return True


def test_backward_compatibility():
    """Test that old JSON format still works"""
    print("\n" + "="*80)
    print("TEST: Backward Compatibility with Old JSON Format")
    print("="*80)
    
    # Old format sample (only 'label', not 'scene_label'/'city_label')
    old_sample = {
        'file': 'old_test.npy',
        'label': 5,
        'domain': 0
    }
    
    dataset = ASCDataset([old_sample], data_root='/tmp', label_key='scene')
    item = dataset[0]
    
    print(f"\nOld format sample: {old_sample}")
    print(f"Converted output:")
    print(f"  scene_label: {item['scene_label'].item()}")
    print(f"  city_label: {item['city_label'].item()}")
    print(f"  domain: {item['domain'].item()}")
    print(f"  split: {item['split']}")
    print(f"  device_id: {item['device_id']}")
    
    # Check defaults
    if item['scene_label'].item() == 5 and item['city_label'].item() == -1:
        print(f"\n✓ TEST PASSED: Old format handled correctly")
        return True
    else:
        print(f"\n✗ TEST FAILED: Unexpected conversion")
        return False


def test_data_leakage_validation():
    """Test data leakage validation"""
    print("\n" + "="*80)
    print("TEST: Data Leakage Validation")
    print("="*80)
    
    # Valid samples (all split='train')
    valid_samples = [
        {'file': 'a.npy', 'split': 'train', 'domain': 0},
        {'file': 'b.npy', 'split': 'train', 'domain': 1},
    ]
    
    print("\nTest 1: Valid samples (all split='train')")
    try:
        validate_no_leakage(valid_samples, allowed_splits=['train'])
        print("  ✓ Validation passed (no exception)")
    except ValueError as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    # Invalid samples (contains split='test')
    invalid_samples = [
        {'file': 'a.npy', 'split': 'train', 'domain': 0},
        {'file': 'b.npy', 'split': 'test', 'domain': 1},  # This should trigger error
    ]
    
    print("\nTest 2: Invalid samples (contains split='test')")
    try:
        validate_no_leakage(invalid_samples, allowed_splits=['train'])
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print("  ✓ Correctly raised ValueError:")
        print(f"    {str(e)[:100]}...")
    
    # Test unseen device validation
    print("\nTest 3: Unseen device validation")
    unseen_samples = [
        {'file': 'a.npy', 'split': 'train', 'domain': 0},
        {'file': 'b.npy', 'split': 'train', 'domain': 6},  # domain 6 = s4 (unseen)
    ]
    
    try:
        validate_no_leakage(unseen_samples, allowed_devices=[0,1,2,3,4,5])
        print("  ✗ Should have raised ValueError for unseen device")
        return False
    except ValueError as e:
        print("  ✓ Correctly raised ValueError for unseen device:")
        print(f"    {str(e)[:100]}...")
    
    print("\n✓ TEST PASSED: Data leakage validation works correctly")
    return True


def test_domain_mask():
    """Test domain mask utilities"""
    print("\n" + "="*80)
    print("TEST: Domain Mask Utilities")
    print("="*80)
    
    import torch
    
    # Create a mock batch
    batch = {
        'features': torch.randn(8, 100, 64),
        'domain': torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),  # Mixed domains
        'scene_label': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
    }
    
    print(f"\nBatch domains: {batch['domain'].tolist()}")
    
    source_mask, target_mask = create_domain_mask(batch, source_domain=0)
    
    print(f"Source mask (domain=0): {source_mask.tolist()}")
    print(f"Target mask (domain!=0): {target_mask.tolist()}")
    print(f"Source count: {source_mask.sum().item()}")
    print(f"Target count: {target_mask.sum().item()}")
    
    # Verify
    if source_mask.sum().item() == 3 and target_mask.sum().item() == 5:
        print("\n✓ TEST PASSED: Domain masks created correctly")
        return True
    else:
        print("\n✗ TEST FAILED: Incorrect mask counts")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("  UNIFIED DATA INTERFACE VALIDATION TESTS")
    print("="*80)
    
    tests = [
        ("Dataset Fields", test_dataset_fields),
        ("Backward Compatibility", test_backward_compatibility),
        ("Data Leakage Validation", test_data_leakage_validation),
        ("Domain Mask", test_domain_mask),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n" + "="*80)
        print("  ALL TESTS PASSED ✓")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("  SOME TESTS FAILED ✗")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
