# 统一数据接口 (Unified Data Interface)

## 概述

本次更新实现了用于教师-学生蒸馏框架的统一数据接口，支持灵活的标签选择和严格的数据泄漏检查。

## 主要变更

### 1. 扩展的数据集字段

每个样本现在返回以下字段：

```python
{
    'features': torch.Tensor,      # [T, F] - 音频特征
    'scene_label': torch.Tensor,   # [] - 场景分类标签 (0-9)
    'city_label': torch.Tensor,    # [] - 城市标签
    'domain': torch.Tensor,        # [] - 域/设备ID (0=A, 1=b, 2=c, ...)
    'split': str,                  # 'train' 或 'test'
    'device_id': str,              # 设备标识符 ('A', 'b', 'c', 's1', ...)
    'path': str,                   # 文件路径（用于缓存键）
    'label': torch.Tensor,         # [] - 主监督标签（根据label_key选择）
}
```

### 2. 灵活的标签选择

通过 `label_key` 参数选择主要监督信号：

```python
# 使用场景标签作为监督
dataloader = get_dataloader(samples, label_key='scene')

# 使用城市标签作为监督
dataloader = get_dataloader(samples, label_key='city')
```

### 3. 向后兼容性

- 支持旧的JSON格式（只有'label'字段）
- 自动将旧的'label'映射到'scene_label'
- 默认值确保现有代码无需修改

### 4. 数据泄漏检查

新增严格的验证函数：

```python
from utils.data_validation import validate_no_leakage

# 检查训练集只包含split='train'
validate_no_leakage(train_samples, allowed_splits=['train'])

# 检查没有使用未见设备 (s4, s5, s6 = domains 6, 7, 8)
validate_no_leakage(train_samples, allowed_devices=[0,1,2,3,4,5])
```

### 5. 域掩码工具

替代硬编码的"前半部分=源域，后半部分=目标域"逻辑：

```python
from utils.domain_mask import create_domain_mask, split_by_domain

# 创建域掩码
source_mask, target_mask = create_domain_mask(batch, source_domain=0)

# 分割批次
source_batch, target_batch = split_by_domain(batch, source_domain=0)
```

## 新增文件

### 核心模块

1. **`utils/data_validation.py`** - 数据泄漏检查
   - `validate_no_leakage()` - 验证split和设备约束
   - `validate_label_availability()` - 验证标签可用性

2. **`utils/domain_mask.py`** - 域掩码工具
   - `create_domain_mask()` - 创建源/目标域布尔掩码
   - `split_by_domain()` - 按域分割批次
   - `get_source_samples()` - 提取源域样本
   - `get_target_samples()` - 提取目标域样本

### 工具脚本

3. **`build_teacher_json.py`** - JSON转换脚本
   - 将旧格式转换为新格式
   - 自动添加缺失字段
   - 创建备份文件

4. **`sanity_check_data.py`** - 数据完整性检查
   - 打印批次字段和形状
   - 显示域分布
   - 验证数据泄漏检查

5. **`test_data_interface.py`** - 单元测试
   - 测试所有新功能
   - 验证向后兼容性

### 修改的文件

6. **`data/dataloader.py`** - 扩展数据加载器
   - `ASCDataset` 添加 `label_key` 参数
   - `__getitem__()` 返回扩展字段
   - `get_dataloader()` 支持 `label_key` 参数

## 使用方法

### 1. 转换现有JSON

如果你的JSON只有旧格式字段：

```bash
python build_teacher_json.py \
    --input data/your_dataset.json \
    --output data/your_dataset_extended.json
```

这会：
- 创建 `your_dataset.json.backup` 备份
- 生成包含所有必需字段的新JSON
- 保持数据完整性

### 2. 运行数据完整性检查

```bash
python sanity_check_data.py \
    --dataset_json data/sample_dataset_extended.json \
    --data_root data/ \
    --label_key scene \
    --batch_size 4
```

输出示例：
```
================================================================================
  Batch Fields and Shapes
================================================================================
Batch fields:
  features       : shape=[4, 100, 64]      dtype=torch.float32
  scene_label    : shape=[4]               dtype=torch.int64
  city_label     : shape=[4]               dtype=torch.int64
  domain         : shape=[4]               dtype=torch.int64
  split          : list with 4 items, first=train
  device_id      : list with 4 items, first=A
  path           : list with 4 items, first=audio_00001.npy
  label          : shape=[4]               dtype=torch.int64

================================================================================
  Domain Distribution in Batch
================================================================================
Domains in batch: {0: 2, 1: 2}
Device IDs: {'A', 'b'}
```

### 3. 在训练脚本中使用

```python
from data.dataloader import get_dataloader
from utils.data_validation import validate_no_leakage
from utils.domain_mask import create_domain_mask

# 加载数据
with open('data/dataset.json') as f:
    dataset = json.load(f)

# 验证训练数据（防止数据泄漏）
train_samples = dataset['train']
validate_no_leakage(
    train_samples, 
    allowed_splits=['train'],
    allowed_devices=[0, 1, 2, 3, 4, 5],  # 排除s4,s5,s6
    phase='training'
)

# 创建dataloader
# 场景教师训练：使用scene标签
scene_loader = get_dataloader(train_samples, label_key='scene')

# 城市教师训练：使用city标签
city_loader = get_dataloader(train_samples, label_key='city')

# 训练循环
for batch in scene_loader:
    # 使用域掩码分离源域和目标域
    source_mask, target_mask = create_domain_mask(batch, source_domain=0)
    
    # 仅对源域计算分类损失
    source_logits = model(batch['features'][source_mask])
    loss = criterion(source_logits, batch['label'][source_mask])
    
    # 域适应损失使用所有样本
    # ...
```

## JSON格式示例

### 完整格式

```json
{
  "train": [
    {
      "file": "audio_00001.npy",
      "scene_label": 0,
      "city_label": 1,
      "domain": 0,
      "split": "train",
      "device_id": "A"
    }
  ],
  "test": [
    {
      "file": "audio_00005.npy",
      "scene_label": 3,
      "city_label": 2,
      "domain": 1,
      "split": "test",
      "device_id": "b"
    }
  ]
}
```

### 设备ID映射

| Domain | Device ID | 描述 |
|--------|-----------|------|
| 0 | A | 源设备 |
| 1 | b | 目标设备1 |
| 2 | c | 目标设备2 |
| 3 | s1 | 目标设备3 |
| 4 | s2 | 目标设备4 |
| 5 | s3 | 目标设备5 |
| 6 | s4 | 未见设备1 |
| 7 | s5 | 未见设备2 |
| 8 | s6 | 未见设备3 |

## 实验协议约束

### 严格规则

1. **源域 (Device A, domain=0)**
   - ✓ 可使用 scene_label 和 city_label
   - ✓ split='train' 用于训练

2. **目标域 (Devices b,c,s1,s2,s3, domains=1-5)**
   - ✓ split='train' 可使用 city_label
   - ✗ split='train' **禁止**使用 scene_label
   - ✗ split='test' **禁止**用于训练

3. **未见设备 (Devices s4,s5,s6, domains=6-8)**
   - ✗ **完全禁止**用于训练
   - ✓ 仅用于最终评估

### 自动检查

所有违规会立即抛出 `ValueError`：

```python
# 示例：试图在训练中使用测试集
validate_no_leakage(samples, allowed_splits=['train'])
# ValueError: DATA LEAKAGE DETECTED in training!
# Sample audio_00005.npy: split='test' not allowed in training (allowed: ['train'])
```

## 运行命令总结

```bash
# 1. 转换JSON（如果需要）
python build_teacher_json.py --input data/your_data.json

# 2. 验证数据接口
python sanity_check_data.py --dataset_json data/your_data.json --data_root data/

# 3. 运行单元测试（仅验证模块）
python -c "from utils.data_validation import validate_no_leakage; print('✓ Import successful')"
```

## 注意事项

1. **不破坏现有代码**：所有新功能都是可选的，默认行为保持不变
2. **域掩码优先**：所有新代码应使用 `create_domain_mask()` 而非假设批次顺序
3. **严格验证**：在训练开始前调用 `validate_no_leakage()`
4. **缓存键**：使用 `batch['path']` 作为教师logits缓存的唯一标识

## 下一步

- Phase 2: 实现三种教师模型
- Phase 3: 添加知识蒸馏损失函数
- Phase 4: 学生训练脚本
- Phase 5: 完整训练流程
