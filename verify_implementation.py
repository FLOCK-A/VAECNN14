#!/usr/bin/env python3
"""Quick verification that the implementation works"""
import sys
sys.path.insert(0, '.')

print("="*80)
print("VERIFYING UNIFIED DATA INTERFACE IMPLEMENTATION")
print("="*80)

# Test 1: Import modules
print("\n[1/4] Testing imports...")
try:
    from utils.data_validation import validate_no_leakage, validate_label_availability
    from utils.domain_mask import create_domain_mask
    print("✅ All utils modules imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Validate data leakage detection
print("\n[2/4] Testing data leakage validation...")
try:
    # Valid case
    valid = [{'file': 'a.npy', 'split': 'train', 'domain': 0}]
    validate_no_leakage(valid, allowed_splits=['train'])
    
    # Invalid case - should raise
    invalid = [{'file': 'b.npy', 'split': 'test', 'domain': 0}]
    try:
        validate_no_leakage(invalid, allowed_splits=['train'])
        print("❌ Should have detected data leakage")
        sys.exit(1)
    except ValueError:
        pass  # Expected
    
    print("✅ Data leakage validation works correctly")
except Exception as e:
    print(f"❌ Validation test failed: {e}")
    sys.exit(1)

# Test 3: Check JSON conversion
print("\n[3/4] Testing JSON conversion...")
try:
    import json
    with open('data/sample_dataset_extended.json') as f:
        data = json.load(f)
    
    required_fields = ['scene_label', 'city_label', 'domain', 'split', 'device_id']
    sample = data['train'][0]
    
    missing = [f for f in required_fields if f not in sample]
    if missing:
        print(f"❌ Missing fields in converted JSON: {missing}")
        sys.exit(1)
    
    print("✅ JSON conversion successful")
    print(f"   Example sample: {sample}")
except Exception as e:
    print(f"❌ JSON conversion test failed: {e}")
    sys.exit(1)

# Test 4: Verify file structure
print("\n[4/4] Verifying file structure...")
import os
required_files = [
    'utils/data_validation.py',
    'utils/domain_mask.py',
    'build_teacher_json.py',
    'sanity_check_data.py',
    'UNIFIED_DATA_INTERFACE.md',
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"❌ Missing files: {missing_files}")
    sys.exit(1)

print("✅ All required files present")

print("\n" + "="*80)
print("✅ ALL VERIFICATION TESTS PASSED")
print("="*80)
print("\nImplementation is ready for Phase 2: Teacher Models")
