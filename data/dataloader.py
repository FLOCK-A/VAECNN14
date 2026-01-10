"""
数据加载器模块
用于加载JSON文件描述的数据集和对应的NPY特征文件
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config.config import config
import os
import random


def worker_init_fn(worker_id):
    """
    DataLoader worker初始化函数，确保多进程环境下随机性独立
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed % 2**32)


class ASCDataset(Dataset):
    """
    声学场景分类数据集 - 支持扩展字段用于教师-学生蒸馏框架
    """

    def __init__(self, samples, data_root=None, transform=None, label_key='scene'):
        """
        初始化数据集
        
        Args:
            samples: 样本列表，包含数据集信息
            data_root: 数据根目录，用于拼接NPY文件路径
            transform: 数据变换
            label_key: str - 'scene' or 'city', 选择哪个标签字段作为主要监督信号
        """
        self.samples = samples
        self.data_root = data_root
        self.transform = transform
        self.label_key = label_key  # 'scene' or 'city'

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, rel_path, root):
        if root is None:
            return rel_path
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(root, rel_path)

    def _maybe_pick_dir_aug(self, sample_info):
        """
        对 domain==DIR_APPLY_DOMAIN 的样本，按概率改为读取离线 DIR 增强 npy。
        """
        rel = sample_info['file']
        domain = sample_info.get('domain', None)

        # 默认走原始
        orig_path = self._resolve_path(rel, self.data_root)

        if not getattr(config, "DIR_ENABLE", False):
            return orig_path

        if domain != getattr(config, "DIR_APPLY_DOMAIN", 0):
            return orig_path

        p = float(getattr(config, "DIR_PROB", 0.0))
        k = int(getattr(config, "DIR_NUM_VARIANTS", 0))
        aug_root = getattr(config, "DIR_AUG_ROOT", None)

        if aug_root is None or k <= 0 or p <= 0:
            return orig_path

        if random.random() >= p:
            return orig_path

        stem, ext = os.path.splitext(rel)
        vidx = random.randrange(k)
        aug_rel = f"{stem}__dir{vidx}{ext}"
        aug_path = self._resolve_path(aug_rel, aug_root)

        # 若离线增强文件缺失则回退
        return aug_path if os.path.exists(aug_path) else orig_path

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: dict with keys:
                - features: torch.Tensor - audio features
                - scene_label: torch.Tensor - scene classification label
                - city_label: torch.Tensor - city label
                - domain: torch.Tensor - domain/device ID
                - split: str - 'train' or 'test'
                - device_id: str - device identifier
                - path: str - file path (for cache key)
                - label: torch.Tensor - main supervision label (scene or city based on label_key)
        """
        sample_info = self.samples[idx]
        
        # 使用DIR增强逻辑选择文件路径
        npy_path = self._maybe_pick_dir_aug(sample_info)
        features = np.load(npy_path)
        
        # 获取各种标签和元信息（支持旧格式向后兼容）
        # 如果JSON中只有'label'字段，假设是scene_label
        scene_label = sample_info.get('scene_label', sample_info.get('label', -1))
        city_label = sample_info.get('city_label', -1)
        domain = sample_info.get('domain', 0)
        split = sample_info.get('split', 'train')  # default to 'train' for backward compatibility
        device_id = sample_info.get('device_id', f'domain_{domain}')
        file_path = sample_info.get('file', '')
        
        # 转换为张量
        features = torch.from_numpy(features).float()
        scene_label_tensor = torch.tensor(scene_label, dtype=torch.long)
        city_label_tensor = torch.tensor(city_label, dtype=torch.long)
        domain_tensor = torch.tensor(domain, dtype=torch.long)
        
        # 根据label_key选择主要监督标签
        if self.label_key == 'city':
            main_label = city_label_tensor
        else:  # default to 'scene'
            main_label = scene_label_tensor
        
        sample = {
            'features': features,
            'scene_label': scene_label_tensor,
            'city_label': city_label_tensor,
            'domain': domain_tensor,
            'split': split,
            'device_id': device_id,
            'path': file_path,
            'label': main_label,  # backward compatibility: main supervision label
        }
        
        if self.transform:
            sample['features'] = self.transform(sample['features'])

        return sample


def get_dataloader(samples, data_root=None, batch_size=None, shuffle=True, num_workers=None, worker_init_fn_py=None, label_key='scene'):
    """
    获取数据加载器
    
    Args:
        samples: 样本列表
        data_root: 数据根目录，用于拼接NPY文件路径
        batch_size: 批大小，默认使用配置文件中的BATCH_SIZE
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数，默认使用配置文件中的NUM_WORKERS
        worker_init_fn_py: worker初始化函数，用于多进程随机性控制
        label_key: str - 'scene' or 'city', 选择主要监督标签字段
        
    Returns:
        dataloader: 数据加载器
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if worker_init_fn_py is None:
        worker_init_fn_py = worker_init_fn
        
    dataset = ASCDataset(samples, data_root, label_key=label_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn_py)
    
    return dataloader