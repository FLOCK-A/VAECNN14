# Teacher Logits Offline Caching

## Overview

`dump_teacher_logits.py` caches teacher model logits offline for faster student training with knowledge distillation.

## Purpose

- Pre-compute teacher logits for training samples
- Save to disk for reuse during student training
- Avoid repeated forward passes through teachers
- Support incremental caching (skip existing files)
- Ensure reproducibility with deterministic execution

## Cache Directory Structure

```
cache/
└── A2{target}/
    ├── city2scene/
    │   ├── {hash1}.npy
    │   ├── {hash2}.npy
    │   └── ...
    ├── scene/
    │   ├── {hash1}.npy
    │   ├── {hash2}.npy
    │   └── ...
    └── cache_index.json
```

### File Naming

- Hash is MD5 hash of the sample file path
- Example: `audio_00001.npy` → `5d41402abc4b2a76b9719d911017c592.npy`
- Collision-free as long as file paths are unique

### cache_index.json

Maps sample file paths to cache files for fast lookup:

```json
{
  "city2scene": {
    "audio_00001.npy": "cache/A2b/city2scene/5d41402abc4b2a76b9719d911017c592.npy",
    "audio_00002.npy": "cache/A2b/city2scene/098f6bcd4621d373cade4e832627b4f6.npy",
    ...
  },
  "scene": {
    "audio_00001.npy": "cache/A2b/scene/5d41402abc4b2a76b9719d911017c592.npy",
    ...
  },
  "num_samples": 1000
}
```

## Usage

### Basic Usage

```bash
python dump_teacher_logits.py \
    --city2scene_teacher_ckpt outputs/teacher_city2scene/A2b/best.pth \
    --scene_teacher_ckpt outputs/teacher_scene/global/best.pth \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b
```

### With All Options

```bash
python dump_teacher_logits.py \
    --city2scene_teacher_ckpt outputs/teacher_city2scene/A2b/best.pth \
    --scene_teacher_ckpt outputs/teacher_scene/global/best.pth \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --cache_root cache \
    --num_scenes 10 \
    --num_cities 12 \
    --use_fp16 \
    --skip_existing \
    --seed 42 \
    --num_verify 10
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--city2scene_teacher_ckpt` | str | Yes | - | Path to City2Scene teacher checkpoint |
| `--scene_teacher_ckpt` | str | Yes | - | Path to Scene teacher checkpoint |
| `--json_path` | str | Yes | - | Path to dataset JSON file |
| `--data_root` | str | Yes | - | Root directory for NPY features |
| `--target_name` | str | Yes | - | Target device (b/c/s1/s2/s3) |
| `--cache_root` | str | No | cache | Root cache directory |
| `--num_scenes` | int | No | 10 | Number of scene classes |
| `--num_cities` | int | No | 12 | Number of city classes |
| `--use_fp16` | flag | No | False | Save logits in fp16 format |
| `--skip_existing` | flag | No | True | Skip caching if file exists |
| `--no_skip_existing` | flag | No | - | Force re-cache all files |
| `--seed` | int | No | 42 | Random seed for reproducibility |
| `--num_verify` | int | No | 10 | Number of samples to verify |

## Features

### 1. Incremental Caching

By default (`--skip_existing`), the script skips files that already exist in cache:

```
Dumping city2scene teacher logits to: cache/A2b/city2scene
Total samples: 1000
Cached: 500, Skipped (existing): 500
```

This allows:
- Resuming interrupted caching runs
- Adding new samples to existing cache
- Efficient re-runs during development

### 2. FP16 Support

Use `--use_fp16` to save logits in half precision:

```bash
python dump_teacher_logits.py ... --use_fp16
```

Benefits:
- 50% disk space savings
- Faster I/O during student training
- Minimal accuracy impact for knowledge distillation

### 3. Reproducibility

Fixed seed and deterministic mode ensure:
- Same logits for same inputs
- Reproducible across runs
- Consistent results for experiments

### 4. Automatic Verification

After caching, the script randomly samples and verifies cache hits:

```
================================================================================
Cache Verification
================================================================================
Verifying 10 random samples...

City2Scene Teacher:
  Hits: 10/10
  Hit Rate: 100.0%

Scene Teacher:
  Hits: 10/10
  Hit Rate: 100.0%

✓ Verification PASSED: 100% hit rate for both teachers
```

## Data Selection

The script caches logits for:
- **Source domain (A)** training samples
- **Target domain** training samples
- **Only split='train'** samples (validates this)

This matches the data used for student training.

## Example Workflow

### Complete Pipeline

```bash
# 1. Train teachers (see TEACHER_TRAINING.md)
python train_teacher_city.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --epochs 50

python train_teacher_city2scene.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --city_teacher_ckpt outputs/teacher_city/A2b/best.pth \
    --epochs 50

python train_teacher_scene.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name global \
    --epochs 50

# 2. Cache teacher logits
python dump_teacher_logits.py \
    --city2scene_teacher_ckpt outputs/teacher_city2scene/A2b/best.pth \
    --scene_teacher_ckpt outputs/teacher_scene/global/best.pth \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b

# 3. Train student with cached logits (Phase 4)
# python train_student.py --use_cached_logits ...
```

### Sanity Test

Quick test with sample data:

```bash
# First, ensure you have teacher checkpoints (train for 1 epoch)
python train_teacher_city.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_city

python train_teacher_city2scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --city_teacher_ckpt outputs/test_city/A2b/best.pth \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_city2scene

python train_teacher_scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name global \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_scene

# Then cache logits
python dump_teacher_logits.py \
    --city2scene_teacher_ckpt outputs/test_city2scene/A2b/best.pth \
    --scene_teacher_ckpt outputs/test_scene/global/best.pth \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --cache_root cache_test \
    --num_verify 5
```

## Reading Cached Logits

### In Student Training

```python
import json
import numpy as np

# Load cache index
with open('cache/A2b/cache_index.json', 'r') as f:
    cache_index = json.load(f)

# Get cached logits for a sample
file_path = 'audio_00001.npy'

# City2Scene teacher logits
city2scene_cache_file = cache_index['city2scene'][file_path]
city2scene_logits = np.load(city2scene_cache_file)  # Shape: [num_scenes]

# Scene teacher logits
scene_cache_file = cache_index['scene'][file_path]
scene_logits = np.load(scene_cache_file)  # Shape: [num_scenes]

# Use in training
# temperature = 4.0
# soft_targets = F.softmax(torch.from_numpy(logits) / temperature, dim=0)
```

### Fast Batch Loading

```python
def load_cached_logits_batch(cache_index, file_paths, teacher='city2scene'):
    """Load cached logits for a batch of samples"""
    logits_list = []
    for path in file_paths:
        cache_file = cache_index[teacher][path]
        logits = np.load(cache_file)
        logits_list.append(logits)
    return np.stack(logits_list)
```

## Cache Management

### Check Cache Size

```bash
du -sh cache/A2b/
# Output: 45M    cache/A2b/
```

### Clear Cache

```bash
# Remove all cached logits
rm -rf cache/A2b/

# Remove only city2scene cache
rm -rf cache/A2b/city2scene/

# Keep cache but force re-cache
python dump_teacher_logits.py ... --no_skip_existing
```

### Verify Cache Integrity

```bash
# Verify specific number of samples
python dump_teacher_logits.py \
    ... \
    --num_verify 100  # Verify 100 random samples
```

## Performance

### Caching Speed

Typical performance on GPU:
- ~10-50 samples/second depending on model size
- 1000 samples: ~20-100 seconds
- 10000 samples: ~3-17 minutes

### Space Usage

Per sample (fp32):
- City2Scene logits: 10 classes × 4 bytes = 40 bytes
- Scene logits: 10 classes × 4 bytes = 40 bytes
- Total: ~80 bytes per sample

For 10,000 samples:
- fp32: ~800 KB
- fp16: ~400 KB

### Student Training Speedup

Benefits:
- No teacher forward pass during student training
- Faster iterations (especially with multiple teachers)
- Reduced GPU memory (no teacher models loaded)

## Error Handling

### Missing Checkpoint

```
Error: [Errno 2] No such file or directory: 'outputs/teacher_city2scene/A2b/best.pth'
```

Solution: Train teachers first (see TEACHER_TRAINING.md)

### Data Leakage Detection

```
ValueError: DATA LEAKAGE DETECTED in Teacher logits caching!
Sample audio_00005.npy: split='test' not allowed in Teacher logits caching (allowed: ['train'])
```

Solution: Ensure only split='train' samples are in dataset

### Hash Collision

Hash collisions are extremely unlikely with MD5 (probability ~10^-38 for reasonable dataset sizes).

If collision occurs, consider using SHA256:
```python
hashlib.sha256(file_path.encode('utf-8')).hexdigest()
```

## Tips

1. **Cache before long experiments**: Cache logits once, use many times
2. **Use fp16 for large datasets**: Saves space with minimal accuracy impact
3. **Verify after first run**: Ensure 100% hit rate
4. **Keep cache_index.json**: Required for fast lookup
5. **Backup cache for important experiments**: Reproducibility insurance

## Limitations

1. **Disk Space**: Large datasets require significant space
2. **Fixed Teachers**: Cache tied to specific teacher checkpoints
3. **No Dynamic Updates**: Re-cache if teachers change
4. **File Path Dependency**: Moving files breaks cache mapping

## Next Steps

After caching logits:
- Phase 4: Implement student training with cached logits
- Phase 5: Add knowledge distillation loss functions
- Phase 6: Support multi-teacher fusion strategies
