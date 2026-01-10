# Teacher Training Scripts Documentation

## Overview

Three independent teacher training scripts for the City2Scene teacher-student distillation framework:

1. **train_teacher_city.py** - Stage-1 City Teacher
2. **train_teacher_city2scene.py** - Stage-2 City2Scene Teacher
3. **train_teacher_scene.py** - Scene Teacher

## 1. City Teacher (Stage-1)

### Purpose
Train a city classification model on source + target training data.

### Data
- Source domain (A) training samples
- Target domain training samples
- Supervision: **city labels only**
- Split: `split=train` only

### Command
```bash
python train_teacher_city.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy/features \
    --target_name b \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/teacher_city \
    --val_ratio 0.1 \
    --num_workers 4
```

### Output
- `outputs/teacher_city/A2b/best.pth` - Best model based on validation accuracy
- `outputs/teacher_city/A2b/last.pth` - Final model after all epochs

### 1 Epoch Sanity Test
```bash
python train_teacher_city.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/test_city \
    --val_ratio 0.2
```

## 2. City2Scene Teacher (Stage-2)

### Purpose
Load frozen City Teacher backbone and train a scene classification head.

### Data
- Source domain (A) training samples **only**
- Supervision: **scene labels**
- Split: `split=train` only

### Key Feature
- **Freezes City Teacher backbone** (feature_extractor)
- Only trains the scene classification head
- Prints trainable parameter count to verify backbone is frozen

### Command
```bash
python train_teacher_city2scene.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy/features \
    --target_name b \
    --city_teacher_ckpt outputs/teacher_city/A2b/best.pth \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/teacher_city2scene \
    --val_ratio 0.1 \
    --num_workers 4 \
    --feature_dim 2048
```

### Output
- `outputs/teacher_city2scene/A2b/best.pth` - Best model (city_model + scene_head)
- `outputs/teacher_city2scene/A2b/last.pth` - Final model

### 1 Epoch Sanity Test
```bash
# First train city teacher for 1 epoch
python train_teacher_city.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_city

# Then train city2scene teacher
python train_teacher_city2scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --city_teacher_ckpt outputs/test_city/A2b/best.pth \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/test_city2scene \
    --val_ratio 0.2
```

### Verification
The script prints:
```
City Teacher (frozen backbone):
  Total parameters: XXXXXX
  Trainable parameters: 0        <-- Should be 0!

Scene Head:
  Total parameters: XXXXXX
  Trainable parameters: XXXXXX

✓ Backbone frozen: 0 trainable params in city_model (should be 0)
```

## 3. Scene Teacher

### Purpose
Train a scene classification model on source domain only.

### Data
- Source domain (A) training samples **only**
- Supervision: **scene labels**
- Split: `split=train` only

### Command
```bash
python train_teacher_scene.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy/features \
    --target_name global \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/teacher_scene \
    --val_ratio 0.1 \
    --num_workers 4
```

### Output
- `outputs/teacher_scene/global/best.pth` - Best model (when target_name='global')
- `outputs/teacher_scene/A2b/best.pth` - Best model (when target_name='b')
- `outputs/teacher_scene/*/last.pth` - Final model

### 1 Epoch Sanity Test
```bash
python train_teacher_scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name global \
    --num_scenes 10 \
    --num_cities 12 \
    --batch_size 4 \
    --epochs 1 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/test_scene \
    --val_ratio 0.2
```

## Unified Arguments

All three scripts share the same argument interface:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--json_path` | str | Yes | - | Path to dataset JSON file |
| `--data_root` | str | Yes | - | Root directory for NPY features |
| `--target_name` | str | Yes* | - | Target device name (b/c/s1/s2/s3) or 'global' |
| `--num_scenes` | int | No | 10 | Number of scene classes |
| `--num_cities` | int | No | 12 | Number of city classes |
| `--batch_size` | int | No | 64 | Batch size |
| `--epochs` | int | No | 50 | Number of training epochs |
| `--lr` | float | No | 1e-3 | Learning rate |
| `--seed` | int | No | 42 | Random seed |
| `--output_dir` | str | No | (varies) | Output directory for checkpoints |
| `--val_ratio` | float | No | 0.1 | Validation split ratio |
| `--num_workers` | int | No | 4 | Number of data loading workers |

*For `train_teacher_scene.py`, target_name can be 'global' for a general model.

**City2Scene specific:**
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--city_teacher_ckpt` | str | Yes | Path to Stage-1 City Teacher checkpoint |
| `--feature_dim` | int | No | Backbone feature dimension (default: 2048) |

## Training/Validation Split

All scripts:
- Use **only `split=train` samples** from JSON
- Filter by domain as needed
- Split train samples internally into train/val using `--val_ratio`
- **Never use `split=test` for training or validation**

## Data Leakage Protection

All scripts include strict validation:
- Checks that only `split=train` samples are used
- Verifies allowed domains (source and/or target)
- Raises `ValueError` if constraints are violated

## Checkpoint Format

### City Teacher & Scene Teacher
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'val_acc': float,
    'val_loss': float,
}
```

### City2Scene Teacher
```python
{
    'epoch': int,
    'city_model_state_dict': OrderedDict,  # Frozen backbone
    'scene_head_state_dict': OrderedDict,   # Trainable head
    'optimizer_state_dict': OrderedDict,
    'val_acc': float,
    'val_loss': float,
}
```

## Example Workflow

### Full Training Pipeline
```bash
# 1. Train City Teacher for target device 'b'
python train_teacher_city.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --epochs 50

# 2. Train City2Scene Teacher using City Teacher
python train_teacher_city2scene.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --city_teacher_ckpt outputs/teacher_city/A2b/best.pth \
    --epochs 50

# 3. Train Scene Teacher (global)
python train_teacher_scene.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name global \
    --epochs 50
```

### Quick Sanity Check (1 epoch each)
```bash
# Test all three teachers with 1 epoch
python train_teacher_city.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/sanity_test_city

python train_teacher_city2scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --city_teacher_ckpt outputs/sanity_test_city/A2b/best.pth \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/sanity_test_city2scene

python train_teacher_scene.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name global \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/sanity_test_scene
```

## Features

✅ **Unified API**: All scripts use the same argument names
✅ **Data Leakage Protection**: Strict validation prevents using test data
✅ **Train/Val Split**: Internal split from `split=train` samples only
✅ **Best/Last Checkpoints**: Saves both best (by val_acc) and final models
✅ **Frozen Backbone Verification**: City2Scene prints trainable param count
✅ **Reproducibility**: Consistent seed setting across all scripts
✅ **Progress Tracking**: Logs train/val acc and loss per epoch

## Files Created

- `train_teacher_city.py` - Stage-1 City Teacher trainer
- `train_teacher_city2scene.py` - Stage-2 City2Scene Teacher trainer
- `train_teacher_scene.py` - Scene Teacher trainer
- `TEACHER_TRAINING.md` - This documentation
