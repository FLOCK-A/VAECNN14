# Student Training with Knowledge Distillation

Comprehensive guide for training student models with knowledge distillation from teacher models.

## Overview

`train_student_distill.py` implements student training with multiple teacher modes and full KD configuration support. Key features:

- **Domain Masking**: CE loss on source samples, KD loss on target samples (configurable)
- **Cached Teacher Logits**: Offline loading from pre-computed cache (no online forward passes)
- **Multiple Teacher Modes**: Single teacher, mean fusion, attention fusion
- **Full KD Configuration**: Temperature, weighting strategies, warmup/ramp scheduling
- **Target Evaluation**: Best model saved based on target test set accuracy

## Teacher Modes

### 1. CE-Only (Baseline)

No knowledge distillation, only cross-entropy loss on source samples.

```bash
python train_student_distill.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode ce_only \
    --batch_size 64 \
    --epochs 50 \
    --output_dir outputs/student
```

**Output**: `outputs/student/A2b/ce_only/best.pth`

### 2. City2Scene Teacher

Use only City2Scene teacher for distillation.

```bash
python train_student_distill.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode city2scene \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --output_dir outputs/student
```

**Output**: `outputs/student/A2b/city2scene/best.pth`

### 3. Scene Teacher

Use only Scene teacher for distillation.

```bash
python train_student_distill.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode scene \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --output_dir outputs/student
```

**Output**: `outputs/student/A2b/scene/best.pth`

### 4. Mean Fusion

Average logits from both City2Scene and Scene teachers.

```bash
python train_student_distill.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode mean_fusion \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --output_dir outputs/student
```

**Fusion Formula**: `fused_logits = (city2scene_logits + scene_logits) / 2`

**Output**: `outputs/student/A2b/mean_fusion/best.pth`

### 5. Attention Fusion

Learnable attention gate to fuse City2Scene and Scene teachers.

```bash
python train_student_distill.py \
    --json_path data/your_dataset.json \
    --data_root /path/to/npy \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode attn_fusion \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --output_dir outputs/student
```

**Fusion Formula**: 
- `γ = AttentionGate(concat(city2scene_logits, scene_logits))`
- `fused_logits = γ * scene_logits + (1-γ) * city2scene_logits`

**Note**: Attention gate is trained jointly with student model.

**Output**: `outputs/student/A2b/attn_fusion/best.pth`

## KD Configuration Parameters

### Temperature (`--temperature`)

Controls softness of probability distributions.
- Higher temperature (e.g., 4.0) → softer distributions, more gradual learning
- Lower temperature (e.g., 1.0) → sharper distributions, closer to hard labels
- **Default**: 4.0
- **Recommended**: 3.0 - 6.0

### KD Alpha (`--kd_alpha`)

Weight for KD loss (1-alpha for CE loss).
- `alpha=0.0` → Only CE loss
- `alpha=1.0` → Only KD loss
- **Default**: 0.7 (70% KD, 30% CE)
- **Recommended**: 0.6 - 0.8

### Loss Type (`--loss_type`)

- `kldiv`: KL Divergence (recommended, default)
- `soft_ce`: Soft Cross-Entropy

### Weighting Strategy (`--kd_weighting`)

How to weight samples in KD loss:
- `none`: All samples weighted equally
- `pmax`: Weight by teacher's max probability (high confidence samples contribute more)
- `threshold`: Binary mask based on confidence threshold

**Example with threshold**:
```bash
--kd_weighting threshold --confidence_threshold 0.7
```
Only samples with teacher confidence > 0.7 receive KD.

### Warmup Epochs (`--warmup_epochs`)

Number of initial epochs with only CE loss (no KD).
- **Default**: 0
- **Recommended**: 5 - 10

### Ramp Epochs (`--kd_ramp_epochs`)

Number of epochs to linearly ramp up KD weight from 0 to target alpha.
- **Default**: 0
- **Recommended**: 5 - 15

**Schedule Example**:
```
--warmup_epochs 5 --kd_ramp_epochs 10 --kd_alpha 0.7
```
- Epochs 0-4: alpha = 0.0 (only CE)
- Epochs 5-14: alpha linearly increases from 0.0 to 0.7
- Epochs 15+: alpha = 0.7 (full KD)

### Apply KD To (`--apply_kd_to`)

Controls which samples receive KD loss:
- `target`: Apply KD only to target domain samples (default, recommended)
- `all`: Apply KD to all samples (source + target)
- `source`: Apply KD only to source domain samples

## Ablation Experiment Configurations

### Config 1: Baseline (CE-Only)

```bash
python train_student_distill.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode ce_only \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/student
```

### Config 2: City2Scene KD

```bash
python train_student_distill.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode city2scene \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --loss_type kldiv \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/student
```

### Config 3: Scene KD

```bash
python train_student_distill.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode scene \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --loss_type kldiv \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/student
```

### Config 4: Mean Fusion

```bash
python train_student_distill.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode mean_fusion \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --loss_type kldiv \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/student
```

### Config 5: Attention Fusion

```bash
python train_student_distill.py \
    --json_path data/dataset.json \
    --data_root data/features \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode attn_fusion \
    --cache_root cache/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --loss_type kldiv \
    --kd_weighting pmax \
    --warmup_epochs 5 \
    --kd_ramp_epochs 10 \
    --batch_size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --seed 42 \
    --output_dir outputs/student
```

## Output Structure

```
outputs/student/
└── A2{target}/
    ├── ce_only/
    │   ├── best.pth
    │   ├── last.pth
    │   ├── config.json
    │   └── history.json
    ├── city2scene/
    │   ├── best.pth
    │   ├── last.pth
    │   ├── config.json
    │   └── history.json
    ├── scene/
    │   ├── best.pth
    │   ├── last.pth
    │   ├── config.json
    │   └── history.json
    ├── mean_fusion/
    │   ├── best.pth
    │   ├── last.pth
    │   ├── config.json
    │   └── history.json
    └── attn_fusion/
        ├── best.pth (includes fusion gate weights)
        ├── last.pth
        ├── config.json
        └── history.json
```

### Checkpoint Contents

**best.pth** (selected by best target test accuracy):
```python
{
    'epoch': 23,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'test_acc': 78.5,
    'val_acc': 76.2,
    'config': {...},
    'fusion_gate_state_dict': ...  # Only for attn_fusion mode
}
```

**history.json**:
```json
[
    {
        "epoch": 1,
        "train_loss": 2.3456,
        "train_acc": 45.67,
        "val_loss": 2.2345,
        "val_acc": 48.23,
        "test_loss": 2.1234,
        "test_acc": 50.12,
        "ce_loss": 1.5678,
        "kd_loss": 0.7778,
        "coverage": 0.85,
        "avg_confidence": 0.72
    },
    ...
]
```

## Training Flow

### 1. Data Loading

- **Training**: Source (A) + Target train split
- **Validation**: Internal split from training data (default 10%)
- **Test**: Target test split (for final evaluation)

### 2. Domain Masking

Each batch contains mixed source/target samples:
- **Source mask**: `domain == 0` (device A)
- **Target mask**: `domain == target_domain`

### 3. Loss Computation

**CE Loss**: Applied only to source samples
```python
if source_mask.sum() > 0:
    ce_loss = F.cross_entropy(logits[source_mask], labels[source_mask])
```

**KD Loss**: Applied to target samples (or all/source, configurable)
```python
teacher_logits = load_from_cache(batch['path'])
kd_loss = compute_kd_loss(student_logits[target_mask], 
                          teacher_logits[target_mask],
                          temperature, kd_alpha)
```

**Combined Loss**:
```python
loss = (1-alpha) * ce_loss + alpha * kd_loss
```

### 4. Evaluation

After each epoch:
1. Validation accuracy (on val split)
2. **Target test accuracy** (on target test set)
3. Save best model if target test accuracy improves

## Sanity Tests

### Quick Test (1 epoch, ce_only)

```bash
python train_student_distill.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode ce_only \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_student
```

### Quick Test (1 epoch, city2scene KD)

```bash
# First ensure cache exists
python dump_teacher_logits.py \
    --city2scene_teacher_ckpt outputs/teacher_city2scene/A2b/best.pth \
    --scene_teacher_ckpt outputs/teacher_scene/global/best.pth \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --cache_root cache_test

# Then train student
python train_student_distill.py \
    --json_path data/sample_dataset_extended.json \
    --data_root data/ \
    --target_name b \
    --num_scenes 10 \
    --teacher_mode city2scene \
    --cache_root cache_test/A2b \
    --temperature 4.0 \
    --kd_alpha 0.7 \
    --batch_size 4 \
    --epochs 1 \
    --output_dir outputs/test_student
```

## Expected Output

```
================================================================================
Student Training with Knowledge Distillation
================================================================================
Target: b (domain=1)
Teacher mode: city2scene
Output directory: outputs/student/A2b/city2scene
Seed: 42
================================================================================
Using device: cuda
Data splits: train=800, val=100, test=200
Loaded cache index from cache/A2b/cache_index.json
Available teachers: ['city2scene', 'scene']
Number of cached samples: 1000

KD Configuration:
  Temperature: 4.0
  KD Alpha: 0.7
  Loss Type: kldiv
  Weighting: pmax
  Warmup Epochs: 5
  Ramp Epochs: 10
  Apply KD to: target

================================================================================
Starting Training
================================================================================

Epoch 1/50: 100%|██████████| 25/25 [00:15<00:00, 1.65it/s, loss=2.1234, acc=48.52%]
Val: 100%|██████████| 4/4 [00:02<00:00, 1.85it/s, loss=2.0123, acc=52.00%]
Test (Target): 100%|██████████| 6/6 [00:03<00:00, 1.92it/s, loss=1.9876, acc=54.50%]

Epoch 1/50
  Train: Loss=2.1234, Acc=48.52%
  Val:   Loss=2.0123, Acc=52.00%
  Test:  Loss=1.9876, Acc=54.50%
  KD Stats: CE=1.4567, KD=0.6667, Coverage=85.23%, Confidence=0.723
  ✓ Best model saved (target acc: 54.50%)

...

================================================================================
Training Completed
================================================================================
Best Target Accuracy: 78.50%
Checkpoints saved to: outputs/student/A2b/city2scene
  - best.pth (target acc: 78.50%)
  - last.pth (final epoch)
  - history.json
================================================================================
```

## Common Issues

### Issue 1: Cache Not Found

**Error**: `FileNotFoundError: Cache index not found`

**Solution**: Run `dump_teacher_logits.py` first to create cache.

### Issue 2: Teacher Mode Mismatch

**Error**: `Teacher mode 'city2scene' not found in cache`

**Solution**: Ensure cache_root contains the required teacher logits. Check `cache_index.json`.

### Issue 3: Domain Imbalance

**Warning**: Very few source or target samples in batch

**Solution**: Increase batch size or check dataset balance.

## Advanced Usage

### Custom Learning Rate Schedule

```python
# Add to train_student_distill.py
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

# In training loop, after optimizer.step():
scheduler.step()
```

### Multi-GPU Training

```python
# Wrap model with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    if fusion_gate is not None:
        fusion_gate = nn.DataParallel(fusion_gate)
```

### Gradient Clipping

```python
# Add before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Next Steps

After training students with different configurations:
1. Compare results across teacher modes
2. Analyze impact of KD parameters (temperature, alpha, weighting)
3. Use `run_all_targets.py` for automated multi-target training
4. Generate summary CSV with all experiment results
