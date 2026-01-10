# VAECNN14 - 基于CNN14的域适应音频分类系统

## 代码功能概述

本项目实现了一个**基于深度学习的域适应音频场景分类系统**，核心功能包括：

1. **音频场景分类**：使用CNN14架构对音频场景进行分类
2. **域适应技术**：支持多种域适应算法，提升模型在不同设备/环境下的泛化能力
3. **DIR离线数据增强**：通过脉冲响应卷积生成增强特征，提高模型鲁棒性
4. **灵活的训练框架**：支持分类和域适应两种训练模式

## 主要功能

### 1. 音频特征提取与分类

**功能描述**：
- 使用预训练的**CNN14**（14层卷积神经网络）作为特征提取器
- 输入：Log-Mel频谱特征（形状：`[batch, time_steps, n_mels]`）
- 输出：音频场景分类结果（10个类别）

**核心文件**：
- `models/feature_extractor.py`: CNN14特征提取器实现
- `models/classifier.py`: 基于CNN14的分类器模型
- `data/dataloader.py`: 数据加载与预处理

**特点**：
- 支持预训练权重加载，加速训练收敛
- 可配置的Dropout率防止过拟合
- 灵活的投影头（Projection Head）用于特征投影

### 2. 域适应算法

**功能描述**：
解决**源域**（训练数据来源设备）和**目标域**（测试数据来源设备）之间的分布差异问题，使模型能够泛化到新设备/环境。

**支持的域适应方法**（在`objectives.py`中实现）：

| 方法名称 | 配置值 | 说明 | 论文出处 |
|---------|--------|------|---------|
| DDC MMD | `ddc` | Deep Domain Confusion - 最小化源域和目标域特征的均值差异 | Deep Domain Confusion (2014) |
| RBF MMD | `mmd` | 基于RBF核的最大均值差异，使用多核技巧 | - |
| CORAL | `coral` | 协方差对齐，最小化二阶统计量差异 | Deep CORAL (2016) |
| Log-CORAL | `log_coral` | 最小熵协方差对齐，在对数空间对齐协方差 | Minimal Entropy Correlation Alignment |
| CMD | `cmd` | 中心矩匹配，匹配高阶中心矩 | Central Moment Discrepancy |
| HoMM | `homm` | 高阶矩匹配，使用2阶或3阶矩 | Higher-order Moment Matching |
| GeoAdapt | `geo_adapt` | 几何适应，使用黎曼几何度量域差异 | - |

**工作原理**：
1. 训练时同时使用源域（有标签）和目标域（无标签）数据
2. 分类损失仅在源域数据上计算
3. 域适应损失在源域和目标域特征上计算，最小化分布差异
4. 总损失 = 分类损失 + λ × 域适应损失

**配置参数**（`config/config.py`）：
```python
EXP_MODE = 'adaptation'          # 'classification' 或 'adaptation'
ADAPT_METHOD = 'homm'            # 选择域适应方法
USER_LAMDA = 0.1                 # 域适应损失权重
HIGHEST_MOMENT = 2               # CMD/HoMM的最高矩阶数
DIST_METRIC_TYPE = 'hilbert'     # GeoAdapt的距离度量类型
```

### 3. DIR（Domain-Invariant Representation）离线增强

**功能描述**：
通过**脉冲响应（IR）卷积**对源域音频进行离线增强，模拟不同的声学环境，提升模型的域不变性。

**工作流程**：

#### 阶段1：离线生成增强特征
```bash
python scripts/offline_build_dir_npy.py \
    --dataset_json "path/to/dataset.json" \
    --wav_root "path/to/wav/files" \
    --out_root "path/to/output/dir" \
    --ir_root "path/to/ir/wavs" \
    --num_variants 8 \
    --n_mels 64 \
    --n_fft 512 \
    --hop 160 \
    --win 400 \
    --fmin 50.0 \
    --fmax 8000.0 \
    --ir_max_len 2047 \
    --fixed_duration 1.0
```

**处理步骤**：
1. 读取原始音频文件（WAV格式）
2. 随机选择IR文件进行卷积操作，模拟房间混响
3. 保持RMS能量一致，避免音量变化
4. 提取Log-Mel特征（与原始特征提取方式完全一致）
5. 应用每文件全局z-score归一化
6. 保存为 `filename__dir{k}.npy` 格式

#### 阶段2：训练时动态选择
- 对源域（domain=0）样本，以70%概率随机选择增强版本
- 目标域样本始终使用原始特征
- 自动回退：增强文件不存在时使用原始特征

**关键技术修复**：
- ✅ IR长度奇偶性：确保IR长度为奇数，避免卷积导致长度变化
- ✅ 音频长度一致性：固定音频时长，确保特征帧数一致
- ✅ 归一化一致性：与原始特征使用相同的z-score归一化
- ✅ Log-Mel计算一致性：使用librosa实现，与原始特征定义完全一致
- ✅ 形状一致性：输出`[T, M]`格式，与原始特征对齐
- ✅ 参数一致性：n_fft、win、fmax等参数与原始特征生成逻辑一致

**配置参数**（`config/config.py`）：
```python
DIR_ENABLE = True                       # 启用DIR增强
DIR_PROB = 0.7                          # 使用增强特征的概率
DIR_NUM_VARIANTS = 8                    # 每个样本的增强变体数
DIR_AUG_ROOT = 'E:\\代码\\data2\\raw_dir'  # 增强特征目录
DIR_APPLY_DOMAIN = 0                    # 仅对源域应用
```

### 4. 训练框架

**功能描述**：
灵活的训练流程，支持预热（Warmup）、余弦退火学习率调度、混合批次训练。

**主流程**（`main.py`）：

```python
# P参数控制训练流程
P = {
    'exp_mode': 'adaptation',      # 实验模式
    'adapt_method': 'homm',        # 域适应方法
    'user_lamda': 0.1,             # 域适应损失权重
    'num_epochs': 100,             # 训练轮数
    'learning_rate': 1e-3,         # 初始学习率
    'warmup_epochs': 5,            # 预热轮数
}
```

**训练特性**：
1. **混合批次训练**：每个批次包含50%源域 + 50%目标域样本
2. **Warmup机制**：前N轮仅进行分类训练，不使用域适应损失
3. **余弦退火**：学习率从初始值按余弦曲线衰减到最小值
4. **最佳模型保存**：根据验证集准确率自动保存最佳模型

**训练输出**：
- 训练损失、分类损失、域适应损失
- 训练/验证准确率
- 最佳模型检查点（`best_model.pth`）
- 最终模型检查点（`final_model.pth`）

## 系统架构

```
VAECNN14/
├── config/
│   └── config.py              # 所有超参数配置
├── models/
│   ├── feature_extractor.py   # CNN14特征提取器
│   └── classifier.py          # 分类器模型
├── data/
│   ├── dataloader.py          # 数据加载器（含DIR增强逻辑）
│   └── sample_dataset.json    # 数据集示例
├── scripts/
│   └── offline_build_dir_npy.py  # DIR增强特征生成脚本
├── main.py                    # 训练主流程
├── objectives.py              # 损失函数（含所有域适应方法）
├── tools.py                   # 工具函数（SPD矩阵、可视化等）
├── README_DIR.md              # DIR功能详细说明
└── DIR_USAGE_GUIDE.md         # DIR使用指南
```

## 技术栈

- **深度学习框架**：PyTorch
- **音频处理**：librosa, soundfile
- **数据格式**：
  - 输入：WAV音频文件
  - 特征：NPY格式的Log-Mel频谱
  - 数据集描述：JSON格式

## 快速开始

### 1. 环境准备
```bash
pip install torch torchaudio numpy librosa soundfile
```

### 2. 准备数据
创建数据集JSON文件（`dataset.json`）：
```json
{
  "train": [
    {"file": "audio_00001.npy", "label": 0, "domain": 0},
    {"file": "audio_00002.npy", "label": 1, "domain": 0}
  ],
  "val": [
    {"file": "audio_00101.npy", "label": 0, "domain": 1},
    {"file": "audio_00102.npy", "label": 1, "domain": 1}
  ],
  "test": [
    {"file": "audio_00201.npy", "label": 0, "domain": 1}
  ]
}
```

### 3. 配置参数
编辑 `config/config.py`，设置数据路径和训练参数。

### 4. （可选）生成DIR增强特征
```bash
python scripts/offline_build_dir_npy.py \
    --dataset_json "path/to/dataset.json" \
    --wav_root "path/to/wavs" \
    --out_root "path/to/output" \
    --ir_root "path/to/irs" \
    --num_variants 8
```

### 5. 开始训练
```bash
python main.py
```

## 实验模式

### 模式1：纯分类
```python
# config/config.py
EXP_MODE = 'classification'
```
- 仅使用分类损失
- 适用于单一设备/环境的数据

### 模式2：域适应
```python
# config/config.py
EXP_MODE = 'adaptation'
ADAPT_METHOD = 'homm'  # 或其他方法
USER_LAMDA = 0.1
```
- 使用分类损失 + 域适应损失
- 适用于跨设备/环境的场景

## 论文实现

本代码实现了以下经典域适应论文的方法：

1. **Deep Domain Confusion** (ICLR 2014)
2. **Deep CORAL** (ECCV 2016)
3. **Central Moment Discrepancy** (CMD)
4. **Higher-order Moment Matching** (HoMM)
5. **Minimal Entropy Correlation Alignment** (Log-CORAL)

## 关键创新点

1. **DIR离线增强**：预生成多版本增强特征，避免训练时实时计算开销
2. **多种域适应方法**：集成7种主流域适应算法，便于对比实验
3. **特征一致性保证**：确保增强特征与原始特征在形状、归一化、参数等方面完全一致
4. **灵活的P参数控制**：通过统一的参数字典控制训练流程，便于实验配置
5. **多进程随机性修复**：解决DataLoader多进程环境下的随机性问题

## 性能优化

- ✅ 预训练权重加载（基于AudioSet的CNN14模型）
- ✅ 混合精度训练支持（可选）
- ✅ 数据并行加载（多进程DataLoader）
- ✅ 学习率预热和余弦退火
- ✅ Dropout正则化防止过拟合
- ✅ 批归一化加速训练收敛

## 应用场景

1. **声学场景分类（ASC）**：识别音频所属的场景（如机场、公园、办公室等）
2. **跨设备音频分析**：训练在设备A上，测试在设备B上
3. **声学环境鲁棒性**：提升模型对不同房间混响的鲁棒性
4. **少样本域适应**：目标域仅有少量或无标签数据的场景

## 参考文档

- **DIR功能详细说明**：`README_DIR.md`
- **DIR使用指南**：`DIR_USAGE_GUIDE.md`

## 许可证

请参考项目许可证文件。

---

**作者**：FLOCK-A  
**最后更新**：2026-01-10
