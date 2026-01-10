# Knowledge Distillation Loss Module

## Overview

`losses/kd.py` provides modular, production-ready knowledge distillation loss functions with temperature scaling, confidence weighting, and scheduling support.

## Features

- ✅ **KL Divergence and Soft Cross-Entropy**: Two distillation loss types
- ✅ **Temperature Scaling**: T^2 gradient compensation
- ✅ **Confidence Weighting**: None, pmax, threshold strategies
- ✅ **Warmup + Ramp Scheduling**: Gradual KD introduction
- ✅ **Domain Masking**: Apply KD selectively to target samples
- ✅ **Comprehensive Statistics**: Coverage, confidence tracking
- ✅ **Numerically Stable**: Handles extreme logit values

## API Reference

### Core Functions

#### `kd_loss(student_logits, teacher_logits, temperature=4.0, loss_type='kldiv')`

Compute knowledge distillation loss between student and teacher.

**Arguments:**
- `student_logits`: Student model logits `[B, num_classes]`
- `teacher_logits`: Teacher model logits `[B, num_classes]` (auto-detached)
- `temperature`: Temperature for softening distributions (default: 4.0)
- `loss_type`: `'kldiv'` (KL divergence) or `'soft_ce'` (soft cross-entropy)
- `reduction`: `'batchmean'`, `'mean'`, `'sum'`, or `'none'`

**Returns:**
- Scalar loss (or per-sample if `reduction='none'`)

**Example:**
```python
from losses.kd import kd_loss

loss = kd_loss(
    student_logits,      # [32, 10]
    teacher_logits,      # [32, 10]
    temperature=4.0,
    loss_type='kldiv'
)
# loss is scalar, ready for backward()
```

#### `compute_confidence_weights(teacher_logits, weighting='none', threshold=0.0)`

Compute per-sample weights based on teacher confidence.

**Arguments:**
- `teacher_logits`: Teacher logits `[B, num_classes]`
- `weighting`: Weighting strategy
  - `'none'`: All weights = 1.0
  - `'pmax'`: Weight by max probability (teacher confidence)
  - `'threshold'`: Binary mask (1 if pmax > threshold, else 0)
- `threshold`: Confidence threshold for `'threshold'` weighting

**Returns:**
- `weights`: Per-sample weights `[B]`
- `stats`: Dict with `coverage`, `avg_confidence`, `min_confidence`, `max_confidence`

**Example:**
```python
from losses.kd import compute_confidence_weights

weights, stats = compute_confidence_weights(
    teacher_logits,
    weighting='threshold',
    threshold=0.7
)
print(f"Coverage: {stats['coverage']:.1%}")  # e.g., 85.5%
print(f"Avg confidence: {stats['avg_confidence']:.3f}")  # e.g., 0.823
```

#### `compute_kd_weight(current_epoch, warmup_epochs=0, kd_ramp_epochs=0, target_alpha=0.7)`

Compute KD loss weight based on training schedule.

**Schedule:**
- Epochs `0` to `warmup_epochs-1`: `kd_alpha = 0.0` (only CE)
- Epochs `warmup_epochs` to `warmup_epochs+kd_ramp_epochs-1`: Linear ramp from 0.0 to `target_alpha`
- Epochs `>= warmup_epochs+kd_ramp_epochs`: `kd_alpha = target_alpha`

**Example:**
```python
from losses.kd import compute_kd_weight

# Warmup 5 epochs, ramp 10 epochs, target alpha 0.7
for epoch in range(20):
    kd_alpha = compute_kd_weight(
        epoch,
        warmup_epochs=5,
        kd_ramp_epochs=10,
        target_alpha=0.7
    )
    print(f"Epoch {epoch}: kd_alpha = {kd_alpha:.3f}")

# Output:
# Epoch 0: kd_alpha = 0.000  (warmup)
# Epoch 5: kd_alpha = 0.000  (start ramp)
# Epoch 10: kd_alpha = 0.350 (mid ramp)
# Epoch 15: kd_alpha = 0.700 (full)
# Epoch 19: kd_alpha = 0.700 (full)
```

#### `compute_student_loss(student_logits, teacher_logits, targets, config, current_epoch=0, domain_mask=None)`

Compute combined student loss (CE + KD) with full scheduling and weighting.

**Arguments:**
- `student_logits`: Student logits `[B, num_classes]`
- `teacher_logits`: Teacher logits `[B, num_classes]`
- `targets`: Ground truth labels `[B]`
- `config`: `KDLossConfig` instance
- `current_epoch`: Current training epoch (for scheduling)
- `domain_mask`: Optional boolean mask `[B]` for target domain samples

**Returns:**
- `loss`: Combined scalar loss
- `stats`: Dict with detailed breakdown
  - `'total_loss'`: Combined loss value
  - `'ce_loss'`: Cross-entropy loss value
  - `'kd_loss'`: KD loss value (0 if in warmup)
  - `'kd_weight'`: Current KD weight (alpha)
  - `'ce_weight'`: Current CE weight (1-alpha)
  - `'coverage'`: Fraction of samples with KD applied
  - `'avg_confidence'`: Average teacher confidence

**Example:**
```python
from losses.kd import compute_student_loss, KDLossConfig

# Create config
config = KDLossConfig(
    temperature=4.0,
    kd_alpha=0.7,
    loss_type='kldiv',
    kd_weighting='pmax',
    confidence_threshold=0.0,
    warmup_epochs=5,
    kd_ramp_epochs=10
)

# Compute loss
loss, stats = compute_student_loss(
    student_logits,
    teacher_logits,
    targets,
    config,
    current_epoch=15,
    domain_mask=target_domain_mask  # Optional
)

# Use in training
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Log statistics
print(f"Epoch 15: Loss={stats['total_loss']:.4f}, "
      f"CE={stats['ce_loss']:.4f}, KD={stats['kd_loss']:.4f}, "
      f"Coverage={stats['coverage']:.1%}")
```

### Configuration

#### `KDLossConfig`

Dataclass for KD loss configuration.

**Fields:**
- `temperature`: Temperature for softening (default: 4.0)
- `kd_alpha`: Weight for KD loss (default: 0.7, CE weight = 1 - kd_alpha)
- `loss_type`: `'kldiv'` or `'soft_ce'` (default: `'kldiv'`)
- `kd_weighting`: `'none'`, `'pmax'`, `'threshold'` (default: `'none'`)
- `confidence_threshold`: Threshold for `'threshold'` weighting (default: 0.0)
- `warmup_epochs`: Epochs with only CE (default: 0)
- `kd_ramp_epochs`: Epochs to ramp up KD weight (default: 0)

**Example:**
```python
from losses.kd import KDLossConfig

# Conservative config (high confidence only)
config = KDLossConfig(
    temperature=3.0,
    kd_alpha=0.5,
    kd_weighting='threshold',
    confidence_threshold=0.8,
    warmup_epochs=10,
    kd_ramp_epochs=20
)

# Aggressive config (all samples, high KD weight)
config = KDLossConfig(
    temperature=5.0,
    kd_alpha=0.9,
    kd_weighting='pmax',
    warmup_epochs=0,
    kd_ramp_epochs=0
)
```

## Usage in Training

### Basic Training Loop

```python
from losses.kd import compute_student_loss, KDLossConfig
import torch.nn.functional as F

# Setup
config = KDLossConfig(
    temperature=4.0,
    kd_alpha=0.7,
    kd_weighting='pmax',
    warmup_epochs=5,
    kd_ramp_epochs=10
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        features = batch['features'].to(device)
        targets = batch['label'].to(device)
        
        # Load cached teacher logits (from dump_teacher_logits.py)
        teacher_logits = load_cached_logits(batch['path'], cache_index)
        teacher_logits = teacher_logits.to(device)
        
        # Student forward pass
        student_logits = student_model(features)
        
        # Compute combined loss
        loss, stats = compute_student_loss(
            student_logits,
            teacher_logits,
            targets,
            config,
            current_epoch=epoch
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}: "
                  f"Loss={stats['total_loss']:.4f}, "
                  f"KD_weight={stats['kd_weight']:.2f}, "
                  f"Coverage={stats['coverage']:.1%}")
```

### With Domain Masking

```python
from utils.domain_mask import create_domain_mask
from losses.kd import compute_student_loss, KDLossConfig

config = KDLossConfig(
    temperature=4.0,
    kd_alpha=0.7,
    kd_weighting='threshold',
    confidence_threshold=0.7
)

for batch in train_loader:
    # Create domain masks
    source_mask, target_mask = create_domain_mask(batch, source_domain=0)
    
    # Forward pass
    student_logits = student_model(batch['features'])
    teacher_logits = load_cached_logits(batch['path'], cache_index)
    
    # Apply KD only to target domain samples
    loss, stats = compute_student_loss(
        student_logits,
        teacher_logits,
        batch['label'],
        config,
        current_epoch=epoch,
        domain_mask=target_mask  # Only target samples get KD
    )
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Multi-Teacher Fusion (Future)

```python
# With mean fusion
city2scene_logits = load_cached_logits(batch['path'], cache_index, teacher='city2scene')
scene_logits = load_cached_logits(batch['path'], cache_index, teacher='scene')

# Mean fusion
teacher_logits = (city2scene_logits + scene_logits) / 2.0

loss, stats = compute_student_loss(
    student_logits,
    teacher_logits,
    targets,
    config,
    current_epoch=epoch
)
```

## Weighting Strategies

### 1. No Weighting (`weighting='none'`)

All samples weighted equally.

```python
config = KDLossConfig(kd_weighting='none')
# All samples contribute equally to KD loss
```

**Use when:**
- Teacher is highly accurate across all samples
- Simple baseline

### 2. Confidence Weighting (`weighting='pmax'`)

Weight samples by teacher's max probability (confidence).

```python
config = KDLossConfig(kd_weighting='pmax')
# High-confidence samples contribute more to KD loss
```

**Use when:**
- Teacher confidence correlates with accuracy
- Want to emphasize reliable predictions

### 3. Threshold Weighting (`weighting='threshold'`)

Binary mask: include only samples above confidence threshold.

```python
config = KDLossConfig(
    kd_weighting='threshold',
    confidence_threshold=0.7
)
# Only samples with teacher confidence > 0.7 get KD loss
```

**Use when:**
- Teacher makes unreliable predictions on some samples
- Want to filter out low-confidence predictions

## Scheduling Strategies

### 1. No Warmup/Ramp (Immediate KD)

```python
config = KDLossConfig(
    warmup_epochs=0,
    kd_ramp_epochs=0,
    kd_alpha=0.7
)
# KD starts immediately at full strength
```

### 2. Warmup Only

```python
config = KDLossConfig(
    warmup_epochs=10,
    kd_ramp_epochs=0,
    kd_alpha=0.7
)
# First 10 epochs: only CE
# Epoch 10+: full KD (alpha=0.7)
```

### 3. Warmup + Gradual Ramp (Recommended)

```python
config = KDLossConfig(
    warmup_epochs=5,
    kd_ramp_epochs=15,
    kd_alpha=0.7
)
# Epochs 0-4: only CE (alpha=0.0)
# Epochs 5-19: linear ramp (alpha: 0.0 → 0.7)
# Epoch 20+: full KD (alpha=0.7)
```

**Recommended for:**
- Stable training
- Avoiding early KD noise
- Smooth transition to distillation

## Testing

### Run Unit Tests

```bash
python test_kd_loss.py
```

### Tests Included

1. **Basic KD Loss**: Validates loss computation
2. **Gradient Flow**: Ensures gradients propagate correctly
3. **Loss Types**: Tests KLDiv vs Soft CE
4. **Confidence Weighting**: Tests all weighting strategies
5. **Scheduling**: Tests warmup + ramp
6. **Combined Loss**: Tests full student loss computation
7. **Domain Masking**: Tests selective KD application
8. **Numerical Stability**: Tests with extreme logit values

### Expected Output

```
================================================================================
KD LOSS UNIT TESTS
================================================================================

================================================================================
Test 1: Basic KD Loss
================================================================================
Student logits shape: torch.Size([8, 10])
Teacher logits shape: torch.Size([8, 10])
KD loss (KLDiv): 0.234567
✓ KD loss is valid (non-negative, no NaN/Inf)

... (more tests)

================================================================================
ALL TESTS PASSED ✓
================================================================================
```

## Advanced Usage

### Custom Loss Function

```python
from losses.kd import kd_loss

def custom_distillation_loss(student_logits, teacher_logits, targets):
    """Custom loss combining CE and multiple KD temperatures"""
    # CE loss
    ce_loss = F.cross_entropy(student_logits, targets)
    
    # KD at multiple temperatures
    kd_loss_t2 = kd_loss(student_logits, teacher_logits, temperature=2.0)
    kd_loss_t4 = kd_loss(student_logits, teacher_logits, temperature=4.0)
    kd_loss_t8 = kd_loss(student_logits, teacher_logits, temperature=8.0)
    
    # Combine
    total_loss = 0.3 * ce_loss + 0.2 * kd_loss_t2 + 0.3 * kd_loss_t4 + 0.2 * kd_loss_t8
    
    return total_loss
```

### Dynamic Temperature

```python
def get_dynamic_temperature(epoch, max_epochs):
    """Increase temperature over training"""
    return 2.0 + 6.0 * (epoch / max_epochs)  # 2.0 → 8.0

for epoch in range(max_epochs):
    temp = get_dynamic_temperature(epoch, max_epochs)
    
    for batch in train_loader:
        loss = kd_loss(
            student_logits,
            teacher_logits,
            temperature=temp
        )
        ...
```

### Per-Class Weighting

```python
from losses.kd import kd_loss

# Compute per-sample KD loss
kd_loss_per_sample = kd_loss(
    student_logits,
    teacher_logits,
    temperature=4.0,
    reduction='none'  # [B]
)

# Get predicted classes
pred_classes = teacher_logits.argmax(dim=1)  # [B]

# Apply per-class weights
class_weights = torch.tensor([1.0, 2.0, 1.5, ...])  # [num_classes]
sample_weights = class_weights[pred_classes]  # [B]

# Weighted loss
weighted_loss = (kd_loss_per_sample * sample_weights).mean()
```

## Tips and Best Practices

### 1. Temperature Selection

- **Low T (1-2)**: Hard targets, preserves peaked distributions
- **Medium T (3-5)**: Balanced, good default
- **High T (6-10)**: Soft targets, transfers more dark knowledge

Recommended: Start with T=4.0, adjust based on validation

### 2. Alpha Selection

- **Low alpha (0.3-0.5)**: Emphasize CE (ground truth)
- **Medium alpha (0.6-0.7)**: Balanced (recommended)
- **High alpha (0.8-0.9)**: Emphasize KD (teacher knowledge)

Recommended: 0.7 for strong teachers, 0.5 for weaker teachers

### 3. Warmup Duration

- No warmup: Risk of early instability
- Short warmup (5-10 epochs): Good baseline
- Long warmup (10-20 epochs): Conservative, stable

Recommended: 5-10 epochs for most cases

### 4. Weighting Strategy

- `'none'`: Default, use if teacher is reliable
- `'pmax'`: Use if teacher confidence varies meaningfully
- `'threshold'`: Use if teacher makes some very poor predictions

Recommended: Start with `'pmax'`, switch to `'threshold'` if needed

### 5. Monitoring

Always log and monitor:
- `coverage`: Fraction of samples receiving KD
- `avg_confidence`: Teacher confidence trend
- `kd_weight`: Current schedule position
- `ce_loss` vs `kd_loss`: Balance check

```python
# Good monitoring
if step % log_interval == 0:
    wandb.log({
        'train/total_loss': stats['total_loss'],
        'train/ce_loss': stats['ce_loss'],
        'train/kd_loss': stats['kd_loss'],
        'train/kd_weight': stats['kd_weight'],
        'train/coverage': stats['coverage'],
        'train/avg_confidence': stats['avg_confidence'],
    })
```

## Integration with Existing Code

### In `train_student.py`

```python
from losses.kd import compute_student_loss, KDLossConfig

# Parse args
parser.add_argument('--temperature', type=float, default=4.0)
parser.add_argument('--kd_alpha', type=float, default=0.7)
parser.add_argument('--kd_weighting', choices=['none', 'pmax', 'threshold'], default='pmax')
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--kd_ramp_epochs', type=int, default=10)

# Create config
config = KDLossConfig(
    temperature=args.temperature,
    kd_alpha=args.kd_alpha,
    kd_weighting=args.kd_weighting,
    warmup_epochs=args.warmup_epochs,
    kd_ramp_epochs=args.kd_ramp_epochs
)

# Use in training loop
for epoch in range(args.epochs):
    for batch in train_loader:
        loss, stats = compute_student_loss(
            student_logits,
            teacher_logits,
            targets,
            config,
            current_epoch=epoch
        )
        ...
```

## Troubleshooting

### Issue: KD loss is NaN

**Causes:**
- Extreme logit values
- Temperature too low

**Solutions:**
```python
# Clip logits before KD
student_logits = torch.clamp(student_logits, -10, 10)
teacher_logits = torch.clamp(teacher_logits, -10, 10)

# Increase temperature
config.temperature = 5.0
```

### Issue: Coverage is 0%

**Causes:**
- Threshold too high
- Teacher confidence very low

**Solutions:**
```python
# Check teacher confidence
weights, stats = compute_confidence_weights(teacher_logits, weighting='none')
print(f"Teacher confidence: {stats['avg_confidence']:.3f}")

# Lower threshold
config.confidence_threshold = 0.5  # Instead of 0.8

# Or use pmax weighting
config.kd_weighting = 'pmax'
```

### Issue: Training unstable

**Causes:**
- No warmup
- Alpha too high
- Temperature too high

**Solutions:**
```python
# Add warmup
config.warmup_epochs = 10
config.kd_ramp_epochs = 20

# Lower alpha
config.kd_alpha = 0.5

# Lower temperature
config.temperature = 3.0
```

## Performance

### Memory Usage

KD loss computation is memory-efficient:
- No large intermediate tensors
- Teacher logits detached (no gradient storage)
- Similar memory to standard CE loss

### Computational Overhead

Minimal overhead vs CE-only training:
- ~5-10% slower per iteration
- Dominated by model forward pass, not loss computation
- Negligible when using cached teacher logits

## References

1. Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
2. Temperature scaling for dark knowledge transfer
3. Confidence-based sample weighting for robust distillation
