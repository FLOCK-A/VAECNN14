# Framework Guardrails and Validation

This document describes the safety checks and validation mechanisms implemented to prevent common errors in the City2Scene teacher-student distillation framework.

## Overview

The framework implements multiple layers of validation to ensure:
1. **No data leakage** from test splits or unseen devices
2. **Proper model freezing** in Stage-2 training
3. **Cache integrity** for student training
4. **End-to-end pipeline correctness**

## 1. Data Leakage Protection

### What it checks:
- Training data does not contain `split='test'` samples
- Training data does not contain unseen devices (s4, s5, s6 / domains 6-8)

### Where it's enforced:
- `utils/data_validation.py::validate_no_leakage()`
- Called in all training scripts:
  - `train_teacher_city.py`
  - `train_teacher_city2scene.py`
  - `train_teacher_scene.py`
  - `train_student_distill.py`

### Example error:

```
================================================================================
🚨 DATA LEAKAGE DETECTED in City Teacher training! 🚨
================================================================================
Found 5 violation(s):

1. Sample audio_001.npy: split='test' not allowed in City Teacher training (allowed: ['train'])
  PATH: data/device_b/test/audio_001.npy [split=test]

2. Sample audio_002.npy: device=s4 (domain=6) not allowed in City Teacher training (allowed domains: [0, 1])
  PATH: data/device_s4/train/audio_002.npy [device=s4, domain=6]

...

================================================================================
⚠️  CRITICAL: Training data contains forbidden samples!
Please ensure:
  - split='test' samples are excluded from training
  - Unseen devices (s4, s5, s6 / domains 6-8) are excluded from training
================================================================================
```

### How to fix:
1. Check your dataset JSON and ensure proper split labels
2. Filter out test samples and unseen devices before training
3. Use `build_teacher_json.py` to convert old format if needed

## 2. Backbone Freeze Verification

### What it checks:
- In Stage-2 City2Scene training, the city backbone is fully frozen
- All `feature_extractor` parameters have `requires_grad=False`
- No gradients flow through the backbone during training

### Where it's enforced:
- `utils/data_validation.py::verify_backbone_frozen()`
- Called in `train_teacher_city2scene.py` after loading City Teacher

### Example error:

```
================================================================================
🚨 BACKBONE NOT FULLY FROZEN in City Teacher! 🚨
================================================================================
Found 127/500 backbone parameters with requires_grad=True:

  1. feature_extractor.layer1.conv1.weight
  2. feature_extractor.layer1.bn1.weight
  3. feature_extractor.layer1.bn1.bias
  4. feature_extractor.layer2.conv1.weight
  5. feature_extractor.layer2.bn1.weight
  ... and 122 more

================================================================================
⚠️  CRITICAL: Backbone should be completely frozen for Stage-2 training!
Please ensure all feature_extractor parameters have requires_grad=False.
================================================================================
```

### How to fix:
1. Check the `freeze_backbone()` function in `train_teacher_city2scene.py`
2. Ensure it freezes all backbone parameters:
   ```python
   for param in city_model.feature_extractor.parameters():
       param.requires_grad = False
   ```
3. Verify the freeze worked by checking trainable parameter count

## 3. Cache Existence Verification

### What it checks:
- All required teacher logits are cached before student training
- Cache index exists at expected location
- Individual cache files exist for all training samples

### Where it's enforced:
- `utils/data_validation.py::verify_cache_exists()`
- `CachedTeacherLogitsLoader.load_logits()` in `train_student_distill.py`

### Example error:

```
================================================================================
🚨 CACHE INDEX NOT FOUND! 🚨
================================================================================
Expected: cache/A2b/cache_index.json

Please run dump_teacher_logits.py to cache teacher logits before student training.
================================================================================
```

Or for missing individual files:

```
================================================================================
🚨 CACHE MISSING FOR 3 SAMPLES! 🚨
================================================================================
Teacher mode: city2scene
Cache root: cache/A2b

Missing cache files:

  1. PATH: data/device_b/train/audio_001.npy (base: audio_001.npy)
  2. PATH: data/device_b/train/audio_002.npy -> cache: cache/A2b/city2scene/abc123.npy (NOT FOUND)
  3. PATH: data/device_A/train/audio_003.npy (base: audio_003.npy)

================================================================================
⚠️  Please run dump_teacher_logits.py to cache all required samples.
================================================================================
```

### How to fix:
1. Run `dump_teacher_logits.py` before student training:
   ```bash
   python dump_teacher_logits.py \
       --city2scene_teacher_ckpt outputs/teacher_city2scene/A2b/best.pth \
       --scene_teacher_ckpt outputs/teacher_scene/global/best.pth \
       --json_path data/dataset.json \
       --data_root data/features \
       --target_name b
   ```
2. Verify cache index exists: `cache/A2b/cache_index.json`
3. Check that cache files are created in `cache/A2b/{city2scene,scene}/`

## 4. Pipeline Sanity Check

### What it tests:
- Complete end-to-end pipeline with minimal data
- All components work correctly without crashing
- Data flows properly through all stages

### Script: `check_pipeline_sanity.py`

### What it runs:
1. Train City Teacher (1 epoch)
2. Train City2Scene Teacher (1 epoch)
3. Train Scene Teacher (1 epoch)
4. Cache teacher logits
5. Train Student with ce_only (1 epoch)
6. Train Student with city2scene KD (1 epoch)

### Usage:

```bash
python check_pipeline_sanity.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --num_scenes 10 \
    --num_cities 12
```

### Example output:

```
================================================================================
Running: Step 1: Train City Teacher
Command: python train_teacher_city.py --json_path data/sample_dataset_extended.json ...
================================================================================
✓ SUCCESS: Step 1: Train City Teacher

================================================================================
Running: Step 2: Train City2Scene Teacher
Command: python train_teacher_city2scene.py --json_path data/sample_dataset_extended.json ...
================================================================================
✓ SUCCESS: Step 2: Train City2Scene Teacher

...

================================================================================
✓ ALL SANITY CHECKS PASSED!
================================================================================

Using temporary directory: /tmp/sanity_check_abc123
You can delete this directory when done inspecting.
```

### When to run:
- Before running full experiments
- After making changes to training scripts
- To verify new dataset is correctly formatted
- To debug pipeline issues

## Best Practices

### 1. Always validate data before training
```python
from utils.data_validation import validate_no_leakage

# For teacher training (exclude test and unseen devices)
validate_no_leakage(
    train_samples,
    allowed_splits=['train'],
    allowed_devices=[0, 1, 2, 3, 4, 5],  # A, b, c, s1, s2, s3
    phase='Teacher training'
)

# For evaluation (can use test split, but not unseen devices)
validate_no_leakage(
    test_samples,
    allowed_splits=['train', 'test'],
    allowed_devices=[0, 1, 2, 3, 4, 5],  # exclude s4, s5, s6
    phase='Evaluation'
)
```

### 2. Verify backbone freeze in Stage-2
```python
from utils.data_validation import verify_backbone_frozen

# After freezing backbone
freeze_backbone(city_model)

# Verify it worked
try:
    verify_backbone_frozen(city_model, model_name='City Teacher')
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

### 3. Check cache before student training
```python
from utils.data_validation import verify_cache_exists

# Get all training file paths
train_paths = [s['path'] for s in train_samples]

# Verify cache exists
try:
    verify_cache_exists(
        cache_root='cache/A2b',
        file_paths=train_paths,
        teacher_mode='city2scene',
        raise_on_missing=True
    )
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

### 4. Run sanity check before big experiments
```bash
# Test pipeline with minimal data first
python check_pipeline_sanity.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/

# If successful, run full experiment
python run_all_targets.py \
    --json_path data/full_dataset.json \
    --data_root data/features \
    --num_scenes 10 \
    --num_cities 12
```

## Error Recovery

### If you see data leakage errors:
1. Check dataset JSON structure
2. Verify split labels are correct ('train' vs 'test')
3. Verify device IDs are correct (0-8)
4. Filter samples before creating dataloaders

### If you see backbone not frozen errors:
1. Check if `freeze_backbone()` was called
2. Verify the function freezes all backbone parameters
3. Check for any code that might re-enable gradients
4. Ensure backbone is in eval mode during training

### If you see cache missing errors:
1. Run `dump_teacher_logits.py` with correct teacher checkpoints
2. Verify cache_index.json was created
3. Check cache directory structure
4. Ensure cache root matches between dumping and loading

### If sanity check fails:
1. Read the error message carefully
2. Check which step failed
3. Run that step manually to see detailed error
4. Fix the issue and re-run sanity check
5. Check temp directory for partial outputs

## Summary

These guardrails ensure:
- ✅ **No accidental data leakage** from test splits or unseen devices
- ✅ **Proper model freezing** in multi-stage training
- ✅ **Cache integrity** for efficient student training
- ✅ **End-to-end correctness** of the complete pipeline

All checks raise informative errors with:
- Clear description of the problem
- Specific file paths that violate constraints
- Actionable suggestions for fixing the issue
- Prevention of silent failures and incorrect results
