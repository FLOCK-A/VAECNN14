# 统一数据接口实现总结

## 实现的功能

✅ Dataset每条样本返回：x, scene, city, domain, split, path
✅ 支持label_key ∈ {scene, city}选择监督字段
✅ 向后兼容现有JSON格式
✅ 严格数据泄漏检查（split=test或unseen进入训练会raise错误）
✅ 域掩码工具（替代batch顺序假设）
✅ JSON转换脚本（不破坏现有数据）
✅ 完整性检查脚本

## 文件变更清单

### 新增文件

1. **`utils/__init__.py`** - Utils模块初始化
2. **`utils/data_validation.py`** - 数据泄漏验证
3. **`utils/domain_mask.py`** - 域掩码工具
4. **`build_teacher_json.py`** - JSON格式转换脚本
5. **`sanity_check_data.py`** - 数据完整性检查脚本
6. **`test_data_interface.py`** - 单元测试脚本
7. **`UNIFIED_DATA_INTERFACE.md`** - 详细文档
8. **`data/sample_dataset_extended.json`** - 转换后的示例数据

### 修改文件

9. **`data/dataloader.py`** - 扩展ASCDataset和get_dataloader

## 运行命令

### 1. 转换现有JSON（如需要）

```bash
# 转换旧格式JSON到新格式（会自动创建备份）
python build_teacher_json.py --input data/your_dataset.json

# 或指定输出路径
python build_teacher_json.py \
    --input data/your_dataset.json \
    --output data/your_dataset_extended.json
```

### 2. 验证数据接口

```bash
# 基础验证（使用示例数据）
python sanity_check_data.py

# 使用你的数据
python sanity_check_data.py \
    --dataset_json data/your_dataset.json \
    --data_root /path/to/npy/files \
    --label_key scene \
    --batch_size 8
```

### 3. 运行单元测试

```bash
# 测试数据验证模块（不需要torch完全安装）
python -c "
from utils.data_validation import validate_no_leakage
samples = [{'file': 'a.npy', 'split': 'train', 'domain': 0}]
validate_no_leakage(samples, allowed_splits=['train'])
print('✓ Validation module works!')
"
```

## 代码示例

### 使用新的数据加载器

```python
from data.dataloader import get_dataloader
from utils.data_validation import validate_no_leakage
from utils.domain_mask import create_domain_mask

# 1. 加载数据集
with open('data/dataset.json') as f:
    dataset = json.load(f)

# 2. 验证数据（防止泄漏）
train_samples = dataset['train']
validate_no_leakage(
    train_samples,
    allowed_splits=['train'],
    allowed_devices=[0,1,2,3,4,5],  # 排除s4,s5,s6
    phase='training'
)

# 3. 创建dataloader
# 场景标签监督
scene_loader = get_dataloader(train_samples, label_key='scene')

# 城市标签监督
city_loader = get_dataloader(train_samples, label_key='city')

# 4. 训练时使用域掩码
for batch in scene_loader:
    # 获取域掩码
    source_mask, target_mask = create_domain_mask(batch, source_domain=0)
    
    # 仅对源域计算分类损失
    source_features = batch['features'][source_mask]
    source_labels = batch['label'][source_mask]
    
    # 训练逻辑
    logits = model(source_features)
    loss = criterion(logits, source_labels)
```

### Batch字段结构

```python
batch = {
    'features': Tensor[B, T, F],     # 音频特征
    'scene_label': Tensor[B],        # 场景标签
    'city_label': Tensor[B],         # 城市标签
    'domain': Tensor[B],             # 域ID
    'split': List[str],              # 'train'或'test'
    'device_id': List[str],          # 设备ID
    'path': List[str],               # 文件路径
    'label': Tensor[B],              # 主监督标签
}
```

## 关键设计决策

1. **向后兼容**：保持`label`字段，根据`label_key`映射
2. **域掩码优先**：替代hardcoded的batch splitting
3. **严格验证**：训练前检查，违规立即报错
4. **最小修改**：只修改必要文件，不破坏现有功能

## 测试验证

已验证功能：
- ✅ 数据验证模块正常工作
- ✅ JSON转换脚本成功运行
- ✅ 扩展字段正确添加
- ✅ 设备ID映射正确
- ✅ Split标记正确

## 下一步

准备进入Phase 2：
- 实现City Teacher
- 实现City2Scene Teacher  
- 实现Scene Teacher
- 创建Teacher Wrapper
