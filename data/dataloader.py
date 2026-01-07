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


class ASCDataset(Dataset):
    """
    声学场景分类数据集
    """

    def __init__(self, samples, data_root=None, transform=None):
        """
        初始化数据集
        
        Args:
            samples: 样本列表，包含数据集信息
            data_root: 数据根目录，用于拼接NPY文件路径
            transform: 数据变换
        """
        self.samples = samples
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 样本数据和标签
        """
        sample_info = self.samples[idx]
        
        # 加载NPY文件
        npy_path = sample_info['file']
        # 如果指定了data_root，则拼接完整路径
        if self.data_root:
            npy_path = os.path.join(self.data_root, npy_path)
        features = np.load(npy_path)
        
        # 获取标签和域信息
        label = sample_info['label']
        domain = sample_info['domain']
        
        # 转换为张量
        features = torch.from_numpy(features).float()
        label = torch.tensor(label, dtype=torch.long)
        domain = torch.tensor(domain, dtype=torch.long)
        
        sample = {
            'features': features,
            'label': label,
            'domain': domain
        }
        
        if self.transform:
            sample['features'] = self.transform(sample['features'])

        return sample


def get_dataloader(samples, data_root=None, batch_size=None, shuffle=True, num_workers=None):
    """
    获取数据加载器
    
    Args:
        samples: 样本列表
        data_root: 数据根目录，用于拼接NPY文件路径
        batch_size: 批大小，默认使用配置文件中的BATCH_SIZE
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数，默认使用配置文件中的NUM_WORKERS
        
    Returns:
        dataloader: 数据加载器
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
        
    dataset = ASCDataset(samples, data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader