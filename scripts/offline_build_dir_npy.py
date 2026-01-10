"""
离线生成DIR（Domain-Invariant Representation）增强特征脚本
用于根据原始音频和IR（脉冲响应）生成增强的log-Mel特征npy文件
"""
import os
import json
import random
import numpy as np
import soundfile as sf
import librosa  # 添加librosa导入，用于与原始脚本一致的log-mel计算


def load_wav_mono(path, target_sr=16000):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def rms_np(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def apply_dir_conv(wav, ir, max_peak=0.99):
    """
    wav, ir: np.float32 1D
    """
    import torch
    import torch.nn.functional as F
    
    wav_t = torch.from_numpy(wav).view(1, 1, -1)   # [1,1,T]
    ir_t  = torch.from_numpy(ir).view(1, 1, -1)    # [1,1,L]
    L = ir_t.shape[-1]
    pad = (L - 1) // 2
    y = F.conv1d(wav_t, ir_t, padding=pad).view(-1).numpy()

    # RMS match
    r0 = rms_np(wav)
    r1 = rms_np(y)
    y = y * (r0 / (r1 + 1e-12))

    # peak limit
    peak = np.max(np.abs(y)) + 1e-12
    if peak > max_peak:
        y = y * (max_peak / peak)
    return y.astype(np.float32)


def compute_log_mel_librosa(y, sr=16000, n_fft=512, hop_length=160, win_length=400, 
                           window="hann", n_mels=64, fmin=50.0, fmax=8000.0, power=2.0, eps=1e-6):
    """
    使用librosa计算log-mel特征，与原始dcase_to_logmel.py完全一致
    返回 [T, n_mels] 格式，与原始脚本一致
    """
    # 计算 mel 频谱图，使用librosa的实现
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window,
        n_mels=n_mels, 
        fmin=fmin,
        fmax=fmax,
        power=power
    )
    # 转换为 log-mel 特征
    log_mel = np.log(mel + eps)
    # 转置为 时间 x n_mels (T, n_mels)，与原始dcase_to_logmel.py一致
    return log_mel.T.astype(np.float32)


def main(
    dataset_json,
    wav_root,
    out_root,
    ir_root,
    apply_domain=0,
    num_variants=8,  # 推荐值改为8
    sr=16000,
    n_fft=512,      # 对齐目标参数
    hop=160,
    win=400,        # 对齐目标参数
    n_mels=64,
    fmin=50.0,
    fmax=8000.0,    # 对齐目标参数
    ir_max_len=2047,  # 改为奇数，避免conv1d导致长度变化
    template_npy=None,
    mean_path=None,
    std_path=None,
    fixed_duration=1.0,  # 对齐目标参数，与DURATION一致
):
    os.makedirs(out_root, exist_ok=True)

    with open(dataset_json, "r", encoding="utf-8") as f:
        info = json.load(f)

    # 只处理 source/train；如果你希望严格按 domain==0，则改成遍历所有 split 按 domain 过滤
    samples = info["train"]

    # IR bank
    ir_paths = [os.path.join(ir_root, p) for p in os.listdir(ir_root) if p.lower().endswith(".wav")]
    assert len(ir_paths) > 0, f"No IR wav in {ir_root}"

    # 读取 template 来对齐 shape（转置与否）
    template_shape = None
    if template_npy is not None and os.path.exists(template_npy):
        template_shape = np.load(template_npy).shape  # e.g. [T,M] or [M,T]

    for s in samples:
        if int(s.get("domain", apply_domain)) != apply_domain:
            continue

        rel_npy = s["file"]
        stem, _ = os.path.splitext(rel_npy)

        # 默认假设 wav 与 npy 同名：audio_00001.npy -> audio_00001.wav
        rel_wav = stem + ".wav"
        wav_path = rel_wav if os.path.isabs(rel_wav) else os.path.join(wav_root, rel_wav)
        if not os.path.exists(wav_path):
            # 如果你的 wav 与 npy 不同目录结构，这里需要你按实际情况改映射规则
            print(f"[WARN] wav not found: {wav_path}, skip")
            continue

        wav = load_wav_mono(wav_path, target_sr=sr)
        
        # 修复 #2：固定音频长度以确保time_steps一致，与dcase_to_logmel.py一致
        if fixed_duration is not None:
            target_len = int(fixed_duration * sr)
            if len(wav) < target_len:
                # 与dcase_to_logmel.py一致：零填充
                wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
            elif len(wav) > target_len:
                # 与dcase_to_logmel.py一致：截断
                wav = wav[:target_len]
        # 如果没有指定fixed_duration，则保持原长度

        for k in range(num_variants):
            ir_path = random.choice(ir_paths)
            ir = load_wav_mono(ir_path, target_sr=sr)

            # IR 预处理：截断 + 确保奇数长度 + L1 归一化（防止能量漂移）
            ir = ir[:min(len(ir), ir_max_len)]
            # 修复 #1：确保 IR 长度为奇数，避免 conv1d 输出长度比输入短 1
            if len(ir) % 2 == 0:
                ir = ir[:-1]
            if len(ir) == 0:  # 防止长度变为0
                continue
            ir = ir / (np.sum(np.abs(ir)) + 1e-8)

            y = apply_dir_conv(wav, ir)

            # 使用与原始dcase_to_logmel.py完全一致的log-mel计算方式
            # 返回 [T, n_mels] 格式，与原始脚本一致
            feat = compute_log_mel_librosa(
                y, 
                sr=sr, 
                n_fft=n_fft, 
                hop_length=hop, 
                win_length=win,
                window="hann",  # 使用参数化的window
                n_mels=n_mels, 
                fmin=fmin,
                fmax=fmax,
                power=2.0,
                eps=1e-6
            )  # [T, n_mels] - 与原始脚本一致

            # per-file global z-score (same as dcase_to_logmel.py)
            # 与dcase_to_logmel.py一致：对单个样本的整张特征图做全局归一化
            m = feat.mean()
            s = feat.std()
            feat = (feat - m) / (s + 1e-6)

            # 对齐 template shape（如果原始是 [M,T]，则转置）
            # 优先保证与template一致，如果没有template则保持[T, M]格式
            if template_shape is not None and feat.shape != template_shape:
                if feat.T.shape == template_shape:
                    feat = feat.T

            out_rel = f"{stem}__dir{k}.npy"
            out_path = out_rel if os.path.isabs(out_rel) else os.path.join(out_root, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, feat.astype(np.float32))

    print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DIR augmented features')
    parser.add_argument('--dataset_json', type=str, required=True, 
                        help='Path to the dataset JSON file')
    parser.add_argument('--wav_root', type=str, required=True,
                        help='Root directory of wav files')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Output directory for augmented npy files')
    parser.add_argument('--ir_root', type=str, required=True,
                        help='Directory containing IR wav files')
    parser.add_argument('--apply_domain', type=int, default=0,
                        help='Domain to apply augmentation (default: 0)')
    parser.add_argument('--num_variants', type=int, default=8,  # 推荐值改为8
                        help='Number of variants per sample (default: 8)')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Target sample rate (default: 16000)')
    parser.add_argument('--n_fft', type=int, default=512,  # 对齐目标参数
                        help='FFT size (default: 512)')
    parser.add_argument('--hop', type=int, default=160,
                        help='Hop length (default: 160)')
    parser.add_argument('--win', type=int, default=400,  # 对齐目标参数
                        help='Window size (default: 400)')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='Number of mel bins (default: 64)')
    parser.add_argument('--fmin', type=float, default=50.0,
                        help='Min frequency for mel filterbank (default: 50.0)')
    parser.add_argument('--fmax', type=float, default=8000.0,  # 对齐目标参数
                        help='Max frequency for mel filterbank (default: 8000.0)')
    parser.add_argument('--ir_max_len', type=int, default=2047,  # 改为奇数
                        help='Max length of IR (default: 2047, odd to avoid conv length mismatch)')
    parser.add_argument('--template_npy', type=str, default=None,
                        help='Template npy file for shape alignment')
    parser.add_argument('--mean_path', type=str, default=None,
                        help='Path to mean.npy for normalization (deprecated, not used)')
    parser.add_argument('--std_path', type=str, default=None,
                        help='Path to std.npy for normalization (deprecated, not used)')
    parser.add_argument('--fixed_duration', type=float, default=1.0,  # 对齐目标参数
                        help='Fix audio duration in seconds (default: 1.0)')

    args = parser.parse_args()
    
    main(
        dataset_json=args.dataset_json,
        wav_root=args.wav_root,
        out_root=args.out_root,
        ir_root=args.ir_root,
        apply_domain=args.apply_domain,
        num_variants=args.num_variants,
        sr=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        win=args.win,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        ir_max_len=args.ir_max_len,
        template_npy=args.template_npy,
        mean_path=args.mean_path,
        std_path=args.std_path,
        fixed_duration=args.fixed_duration
    )