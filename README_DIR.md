# DIR (Domain-Invariant Representation) 离线增强功能

## 概述

本项目实现了DIR（Domain-Invariant Representation）离线增强功能，用于在训练过程中对源域数据进行动态增强，以提高模型的域适应能力。

## 功能特点

1. **离线预生成**：预先生成多个增强版本的特征文件，避免训练时实时计算
2. **概率随机选择**：训练时按设定概率随机选择原始特征或增强特征
3. **目标域不变**：仅对源域（domain=0）应用增强，目标域保持不变
4. **无缝集成**：无需修改训练主流程，自动适配现有代码结构
5. **多进程随机性修复**：解决了DataLoader多进程环境下的随机性问题
6. **长度一致性保证**：确保生成的特征与原始特征形状完全一致
7. **特征定义一致性**：使用与原始特征相同的librosa实现计算log-mel特征
8. **形状一致性保证**：确保输出特征形状与原始特征一致（[T, M]格式）
9. **参数一致性保证**：确保Mel参数与原始特征生成逻辑完全一致

## 配置参数

在 `config/config.py` 中新增了以下参数：

- `DIR_ENABLE`: 是否启用DIR增强功能（默认：True）
- `DIR_PROB`: 对源域样本应用增强的概率（默认：0.7）
- `DIR_NUM_VARIANTS`: 每个样本预生成的增强版本数（默认：8）
- `DIR_AUG_ROOT`: 存放增强特征文件的根目录（默认：'E:\代码\data2\raw_dir'）
- `DIR_APPLY_DOMAIN`: 应用增强的域编号（默认：0，即源域）

## 目录结构约定

- 原始特征目录：`DATA_ROOT` (如 `E:\代码\data2\raw`)
- 增强特征目录：`DIR_AUG_ROOT` (如 `E:\代码\data2\raw_dir`)

增强特征文件命名规则：`{原始文件名}__dir{k}.npy`
例如：
- 原始：`audio_00001.npy`
- 增强：`audio_00001__dir0.npy`, `audio_00001__dir1.npy`, 等

## 离线生成脚本

使用以下命令生成DIR增强特征：

```bash
python scripts/offline_build_dir_npy.py \
    --dataset_json "path/to/dataset.json" \
    --wav_root "path/to/wav/files" \
    --out_root "path/to/output/dir" \
    --ir_root "path/to/ir/wavs" \
    --num_variants 8 \
    --n_mels 64 \
    --n_fft 512 \  # 与原始特征生成逻辑一致
    --hop 160 \
    --win 400 \  # 与原始特征生成逻辑一致
    --fmin 50.0 \
    --fmax 8000.0 \  # 与原始特征生成逻辑一致
    --ir_max_len 2047 \  # 确保为奇数，避免卷积长度不匹配
    --fixed_duration 1.0 \  # 与原始特征生成逻辑一致
    --template_npy "path/to/template.npy"
```

## 关键修复说明

### 修复 #1：IR长度奇偶性问题
- **问题**：当IR长度为偶数时，卷积操作会导致输出长度比输入短1个采样点，进而影响STFT帧数
- **解决方案**：确保IR长度为奇数（默认`ir_max_len`从2048改为2047）
- **代码**：`if len(ir) % 2 == 0: ir = ir[:-1]`

### 修复 #2：音频长度一致性问题
- **问题**：不同长度的音频会产生不同帧数的特征，导致训练时堆叠错误
- **解决方案**：添加`--fixed_duration`参数，将音频统一裁剪或填充到固定长度
- **代码**：固定长度填充/裁剪机制

### 修复 #3：归一化一致性问题
- **问题**：增强特征与原始特征使用不同的归一化方式，导致训练时混入两套统计尺度
- **解决方案**：使用与原始特征相同的**每文件全局z-score归一化**：
  ```python
  m = feat.mean()
  s = feat.std()
  feat = (feat - m) / (s + 1e-6)
  ```

### 修复 #4：log-mel计算一致性问题
- **问题**：增强特征与原始特征使用不同的log-mel计算方式，导致特征定义不一致
- **解决方案**：使用与原始特征相同的librosa实现计算log-mel特征：
  ```python
  mel = librosa.feature.melspectrogram(...)
  log_mel = np.log(mel + eps)
  ```

### 修复 #5：特征形状一致性问题
- **问题**：增强特征与原始特征使用不同的形状约定（[M, T] vs [T, M]）
- **解决方案**：确保输出形状与原始特征一致（[T, M]格式），与原始脚本一致：
  ```python
  # 转置为 时间 x n_mels (T, n_mels)，与原始dcase_to_logmel.py一致
  return log_mel.T.astype(np.float32)
  ```

### 修复 #6：参数一致性问题
- **问题**：增强特征与原始特征使用不同的Mel参数（如n_fft, win, fmax等）
- **解决方案**：确保使用与原始特征相同的参数：
  - `n_fft=512` (与原始特征生成逻辑一致)
  - `win_length=400` (与原始特征生成逻辑一致)
  - `fmax=8000.0` (与原始特征生成逻辑一致)
  - `eps=1e-6` (与原始特征生成逻辑一致)
  - `fixed_duration=1.0` (与原始特征生成逻辑一致)

### 优化：预计算加速
- **优化**：预计算Hann窗函数和Mel滤波器组，避免重复计算，提高生成效率

## 特征一致性保证

为确保生成的增强特征与原始特征保持一致，请注意以下几点：

1. **形状一致**：输出特征形状需与原始特征相同（当前为 [101, 64]）
2. **参数一致**：Mel频谱提取参数需与原始特征提取时保持一致（n_fft=512, win=400, fmax=8000.0等）
3. **归一化一致**：使用与原始特征相同的每文件全局z-score归一化
4. **特征定义一致**：使用与原始特征相同的librosa实现计算log-mel特征
5. **形状格式一致**：确保输出为[T, M]格式与原始特征一致
6. **音频长度一致**：使用固定的音频长度（默认1.0秒）确保time_steps一致

## 训练时行为

在训练过程中：
- 对于domain=0（源域）的样本：按`DIR_PROB`概率随机选择原始特征或某个增强版本
- 对于domain=1（目标域）的样本：始终使用原始特征
- 如果增强特征文件不存在，自动回退到原始特征

## 多进程随机性修复

为解决DataLoader在多进程环境下的随机性问题，我们在DataLoader中添加了`worker_init_fn`函数，确保：
- 每个worker使用独立的随机种子
- 随机性不会重复，增强效果更佳
- 训练结果更加稳定可靠

## 使用建议

1. **存储考虑**：`DIR_NUM_VARIANTS`不宜过大，通常8个变体已足够
2. **概率设置**：`DIR_PROB`推荐设置为0.5-0.8之间
3. **IR多样性**：IR（脉冲响应）文件应具有足够的多样性以获得更好的增强效果
4. **多进程训练**：使用多进程DataLoader时，确保设置了正确的worker_init_fn
5. **长度一致性**：使用`--fixed_duration`参数确保音频长度一致，避免time_steps不匹配
6. **特征一致性**：确保增强特征与原始特征在形状、参数、归一化和特征定义方面完全一致
7. **参数对齐**：确保Mel参数与原始特征生成逻辑完全一致（n_fft=512, win=400, fmax=8000.0等）

## 验证

可通过运行以下命令验证特征一致性：

```bash
python -c "
import numpy as np
# 检查形状和归一化一致性
raw_feat = np.load('path/to/original.npy')
dir_feat = np.load('path/to/dir_enhanced.npy')
print(f'原始特征形状: {raw_feat.shape}, 均值: {raw_feat.mean():.6f}, 标准差: {raw_feat.std():.6f}')
print(f'增强特征形状: {dir_feat.shape}, 均值: {dir_feat.mean():.6f}, 标准差: {dir_feat.std():.6f}')
"
```