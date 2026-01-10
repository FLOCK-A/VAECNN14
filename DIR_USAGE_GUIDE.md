# DIR增强功能使用指南

## 概述

本指南介绍如何使用DIR（Domain-Invariant Representation）离线增强功能，该功能通过在训练时随机选择原始特征或增强特征来提升模型的域适应能力。

## 完整工作流程

### 1. 准备阶段
在开始使用DIR增强功能之前，您需要准备以下内容：

- **WAV音频文件**：原始音频文件，用于生成增强特征
- **IR（脉冲响应）文件**：66个WAV格式的IR文件，用于模拟不同的声学环境
- **模板特征文件**：一个示例npy文件，确保生成的增强特征形状与原始特征一致

### 2. 离线生成增强特征

使用以下命令生成增强特征：

```bash
# 生成增强特征
python scripts/offline_build_dir_npy.py \
    --dataset_json "E:\代码\data2\dann_dataset_b.json" \
    --wav_root "E:\path\to\your\wav\files" \
    --out_root "E:\代码\data2\raw_dir" \
    --ir_root "E:\path\to\your\ir\files" \
    --num_variants 8 \
    --n_mels 64 \
    --n_fft 512 \  # 与原始特征生成逻辑一致
    --hop 160 \
    --win 400 \  # 与原始特征生成逻辑一致
    --fmin 50.0 \
    --fmax 8000.0 \  # 与原始特征生成逻辑一致
    --ir_max_len 2047 \  # 确保为奇数，避免卷积长度不匹配
    --fixed_duration 1.0 \  # 与原始特征生成逻辑一致
    --template_npy "E:\path\to\template.npy"
```

参数说明：
- `--dataset_json`: 数据集JSON文件路径
- `--wav_root`: WAV音频文件根目录
- `--out_root`: 输出增强特征文件目录
- `--ir_root`: IR文件根目录（需要66个WAV文件）
- `--num_variants`: 每个样本生成的增强变体数量（推荐8个）
- `--n_fft`: FFT大小（默认512，与原始特征生成逻辑一致）
- `--win`: 窗口大小（默认400，与原始特征生成逻辑一致）
- `--fmax`: Mel滤波器最大频率（默认8000.0，与原始特征生成逻辑一致）
- `--ir_max_len`: IR最大长度（建议使用奇数如2047，避免卷积导致长度变化）
- `--fixed_duration`: 固定音频时长（秒），确保time_steps一致（默认1.0秒，与原始特征生成逻辑一致）
- `--template_npy`: 模板npy文件路径（用于确保形状一致）
- `--num_variants`: 每个样本的增强变体数量（推荐8个）

### 3. 验证增强特征

生成完成后，验证增强特征是否正确创建：

```bash
# 检查增强特征文件
python -c "
import numpy as np
import os
# 检查原始和增强特征的形状和归一化
raw_path = 'E:/path/to/your/raw/file.npy'  # 替换为实际路径
dir_path = 'E:/path/to/your/dir/file__dir0.npy'  # 替换为实际路径
if os.path.exists(raw_path):
    raw_feat = np.load(raw_path)
    print(f'原始特征形状: {raw_feat.shape}')
    print(f'原始特征均值: {raw_feat.mean():.6f}, 标准差: {raw_feat.std():.6f}')
if os.path.exists(dir_path):
    dir_feat = np.load(dir_path)
    print(f'增强特征形状: {dir_feat.shape}')
    print(f'增强特征均值: {dir_feat.mean():.6f}, 标准差: {dir_feat.std():.6f}')
"
```

### 4. 开始训练

生成增强特征后，您可以像往常一样训练模型，DataLoader会自动按配置的概率随机选择原始特征或增强特征：

```bash
# 正常训练，DataLoader会自动应用DIR增强
python main.py
```

## 配置参数详解

在 `config/config.py` 中的DIR相关参数：

- `DIR_ENABLE`: 是否启用DIR增强（默认True）
- `DIR_PROB`: 对源域样本应用增强的概率（默认0.7）
- `DIR_NUM_VARIANTS`: 每个样本的增强变体数量（默认8）
- `DIR_AUG_ROOT`: 增强特征文件根目录
- `DIR_APPLY_DOMAIN`: 应用增强的域（默认0，即源域）

## 工作原理

### 离线生成阶段
1. 从JSON文件中筛选domain==0的样本
2. 对每个样本，随机选择IR文件进行卷积操作
3. 将增强后的音频转换为log-Mel特征，**使用与原始特征完全一致的librosa实现，输出[T, M]格式**
4. 对每个特征应用**每文件全局z-score归一化**（与原始特征处理方式一致）
5. 保存为 `filename__dir{k}.npy` 格式的增强特征

### 训练阶段
1. 对于domain==0的样本，以`DIR_PROB`概率随机选择
2. 随机选择原始特征或某个增强变体（`__dir0` 到 `__dir{N-1}`）
3. 如果增强文件不存在，自动回退到原始特征
4. 其他域的样本始终使用原始特征

## 重要修复说明

### 修复 #1：IR长度奇偶性问题
- **问题**：当IR长度为偶数时，卷积操作会导致输出长度比输入短1个采样点，进而影响STFT帧数
- **解决方案**：确保IR长度为奇数（默认`ir_max_len`从2048改为2047）

### 修复 #2：音频长度一致性问题
- **问题**：不同长度的音频会产生不同帧数的特征，导致训练时堆叠错误
- **解决方案**：添加`--fixed_duration`参数，将音频统一裁剪或填充到固定长度，与原始特征处理方式一致

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

## 验证和调试

使用以下脚本验证功能：

```bash
# 验证特征一致性
python -c "
import numpy as np
# 检查形状和归一化一致性
raw_feat = np.load('path/to/original.npy')
dir_feat = np.load('path/to/dir_enhanced.npy')
print(f'原始特征形状: {raw_feat.shape}, 均值: {raw_feat.mean():.6f}, 标准差: {raw_feat.std():.6f}')
print(f'增强特征形状: {dir_feat.shape}, 均值: {dir_feat.mean():.6f}, 标准差: {dir_feat.std():.6f}')
print(f'形状一致: {raw_feat.shape == dir_feat.shape}')
print(f'归一化一致 (均值接近0): {abs(raw_feat.mean()) < 1e-3 and abs(dir_feat.mean()) < 1e-3}')
print(f'归一化一致 (标准差接近1): {abs(raw_feat.std()-1) < 1e-3 and abs(dir_feat.std()-1) < 1e-3}')
"
```

## 注意事项

1. **存储空间**：增强特征会占用额外存储空间，约为原始特征的N倍（N为变体数量）
2. **IR多样性**：IR文件应具有足够的多样性以获得良好的增强效果
3. **特征一致性**：确保增强特征与原始特征在形状、参数和归一化方面完全一致
4. **多进程训练**：已修复多进程环境下的随机性问题
5. **概率设置**：推荐`DIR_PROB`设置为0.5-0.8之间
6. **长度一致性**：使用`--fixed_duration`参数确保音频长度一致，避免time_steps不匹配
7. **归一化一致性**：增强特征使用与原始特征相同的每文件全局z-score归一化
8. **log-mel一致性**：增强特征使用与原始特征相同的librosa实现计算log-mel特征
9. **形状一致性**：增强特征使用与原始特征相同的[T, M]形状格式
10. **参数一致性**：增强特征使用与原始特征相同的Mel参数（n_fft, win, fmax等）
11. **推荐参数**：`DIR_NUM_VARIANTS=8`, `DIR_PROB=0.7`

## 故障排除

如果遇到问题，请检查：

1. 确认增强特征目录存在
2. 确认增强特征文件按正确命名规则生成
3. 确认特征形状与原始特征一致（特别是time_steps维度）
4. 检查JSON文件中的domain标签是否正确
5. 验证IR文件长度是否为奇数（或使用默认的2047）
6. 确认音频文件长度是否一致（或使用`--fixed_duration`参数）
7. 验证归一化是否一致（均值接近0，标准差接近1）
8. 确认log-mel计算方式与原始特征一致（使用librosa实现）
9. 确认特征形状与原始特征一致（[T, M]格式）
10. 确认Mel参数与原始特征一致（n_fft=512, win=400, fmax=8000.0等）