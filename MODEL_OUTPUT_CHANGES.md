# Model Output Changes: Raw Logits + Forward Features

## Summary of Changes

### Modified Files
- `models/classifier.py` - Updated `Cnn14Classifier` class

### Changes Made

#### 1. Removed LogSoftmax from Classifier Output

**Before:**
```python
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, 1024),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1024, classes_num),
    nn.LogSoftmax(dim=1)  # ← REMOVED
)
```

**After:**
```python
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, 1024),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(1024, classes_num)
    # 注意：不使用 LogSoftmax，输出 raw logits 供 cross_entropy 使用
)
```

**Why:** 
- `torch.nn.functional.cross_entropy` expects raw logits, not log probabilities
- Already used correctly in `objectives.py` (line 340, 345, 396)
- Removing LogSoftmax makes the API clearer and more standard

#### 2. Added `forward_features()` Method

```python
def forward_features(self, input_data):
    """
    提取 backbone 特征（用于 Stage-2 City2Scene Teacher）
    
    Args:
        input_data: 输入数据
        
    Returns:
        features: torch.Tensor - backbone embedding [B, feature_dim]
    """
    features = self.feature_extractor(input_data)
    features = features.view(features.size(0), -1)  # 展平特征
    return features
```

**Purpose:**
- Allows Stage-2 City2Scene Teacher to extract frozen city backbone features
- Provides clean API for feature extraction without classification head

#### 3. Added `extract_features()` Alias

```python
def extract_features(self, input_data):
    """
    提取 backbone 特征的别名方法
    
    Args:
        input_data: 输入数据
        
    Returns:
        features: torch.Tensor - backbone embedding [B, feature_dim]
    """
    return self.forward_features(input_data)
```

**Purpose:**
- Alternative naming convention for feature extraction
- Provides flexibility in API usage

#### 4. Updated `forward()` Documentation

- Clarified that output is **raw logits**, not log probabilities
- Added note about compatibility with `cross_entropy`

### API Compatibility

✅ **No Breaking Changes**
- `forward()` signature unchanged
- `return_adapt_features` parameter still works
- `objectives.py` already uses `cross_entropy` with logits - no changes needed
- `main.py` uses model via `objectives.compute_batch_loss` - no changes needed

### Cross Entropy Usage

`torch.nn.functional.cross_entropy` works directly with raw logits:

```python
# In objectives.py (already correct)
class_loss = torch.nn.functional.cross_entropy(
    batch['logits'],  # raw logits from model
    batch['labels']   # target labels
)
```

The function internally applies:
1. LogSoftmax to convert logits to log probabilities
2. NLLLoss to compute negative log likelihood

So removing LogSoftmax from the model is the correct approach.

### Stage-2 Training Pattern

For City2Scene Teacher:

```python
# Load Stage-1 City Teacher
city_model = Cnn14Classifier(classes_num=num_cities)
city_model.load_state_dict(torch.load('city_teacher.pth'))

# Freeze city backbone
for param in city_model.feature_extractor.parameters():
    param.requires_grad = False

# Create scene head
scene_head = nn.Linear(feature_dim, num_scenes)

# Training loop
city_model.eval()
scene_head.train()

for batch in dataloader:
    # Extract frozen features
    with torch.no_grad():
        features = city_model.forward_features(batch['features'])
    
    # Train scene head
    scene_logits = scene_head(features.detach())
    loss = F.cross_entropy(scene_logits, batch['scene_label'])
    loss.backward()  # Only updates scene_head
```

## Testing

Run unit tests:
```bash
python test_model_logits.py
```

Tests verify:
- ✅ Model outputs raw logits with correct shape
- ✅ No NaN values in output
- ✅ `cross_entropy` works directly with logits
- ✅ Gradients computed correctly
- ✅ `forward_features()` returns correct backbone embeddings
- ✅ `extract_features()` is equivalent to `forward_features()`
- ✅ Stage-2 training pattern works (frozen backbone + trainable head)

## Migration Guide

### For Existing Code

**No changes needed** if using through `objectives.compute_batch_loss()`:
```python
# This still works exactly the same
batch_for_loss = {
    'logits': model(features),
    'labels': labels
}
batch_for_loss = compute_batch_loss(batch_for_loss, P)
loss = batch_for_loss['loss_tensor']
```

### For New Teacher Training Scripts

```python
# City Teacher
model = Cnn14Classifier(classes_num=num_cities)
logits = model(features)  # raw logits
loss = F.cross_entropy(logits, city_labels)

# City2Scene Teacher (Stage-2)
city_model.eval()
with torch.no_grad():
    features = city_model.forward_features(audio_input)
scene_logits = scene_head(features)
loss = F.cross_entropy(scene_logits, scene_labels)
```

## Files Modified

```
models/classifier.py     Modified: Removed LogSoftmax, added forward_features()
test_model_logits.py     Created:  Unit tests for model changes
MODEL_OUTPUT_CHANGES.md  Created:  This documentation
```

## Verification

All changes verified through unit tests:
- Raw logits output validated
- Cross entropy compatibility confirmed
- Feature extraction tested
- Stage-2 training pattern validated
- No breaking changes to existing code
