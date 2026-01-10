"""
基于Cnn14LogMel的分类器模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .feature_extractor import Cnn14LogMel
from config.config import config


class Cnn14Classifier(nn.Module):
    """
    基于Cnn14LogMel的分类器模型
    """
    def __init__(self, classes_num=10, checkpoint_path=None, use_projection_head=False, dropout=None,adapt_dim=128,use_adapt_head=True):
        """
        初始化分类器
        
        Args:
            classes_num: 分类数量
            checkpoint_path: 预训练权重路径
            use_projection_head: 是否使用投影头
            dropout: Dropout比例，默认使用配置文件中的DROPOUT
        """
        super(Cnn14Classifier, self).__init__()
        
        if dropout is None:
            dropout = config.DROPOUT
        
        # 使用Cnn14LogMel作为特征提取器
        self.feature_extractor = Cnn14LogMel(
            classes_num=classes_num,
            checkpoint_path=checkpoint_path,
            use_projection_head=use_projection_head
        )
        
        # 分类头 - 基于特征维度进行分类
        self.dropout = nn.Dropout(dropout)
        
        # 获取特征维度 (假设为2048，需根据实际CNN14输出调整)
        feature_dim = config.FEATURE_DIM
        
        # 标签分类器 - 输出 raw logits（不使用 LogSoftmax）
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
        self.use_adapt_head = use_adapt_head and (adapt_dim is not None) and (adapt_dim > 0)
        if self.use_adapt_head:
            self.adapt_layer = nn.Sequential(
                nn.Linear(feature_dim, 1024),
                nn.Linear(1024, 512),
                nn.Linear(512, adapt_dim),
                nn.BatchNorm1d(adapt_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.adapt_layer = None

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
    
    def extract_features(self, input_data):
        """
        提取 backbone 特征的别名方法
        
        Args:
            input_data: 输入数据
            
        Returns:
            features: torch.Tensor - backbone embedding [B, feature_dim]
        """
        return self.forward_features(input_data)
    
    def forward(self, input_data, return_adapt_features=False):
        """
        前向传播
        
        Args:
            input_data: 输入数据
            return_adapt_features: 是否返回用于域适应的特征

        Returns:
            如果return_adapt_features=False: raw logits [B, classes_num]
            如果return_adapt_features=True: (raw logits, 用于域适应的特征)
            
        注意：输出为 raw logits，不包含 LogSoftmax，可直接用于 cross_entropy
        """
        # 提取特征
        features = self.forward_features(input_data)
        features = self.dropout(features)
        
        # 分类器输出 raw logits
        output = self.classifier(features)
        
        if return_adapt_features:
            # 返回用于域适应的特征
            if hasattr(self, 'adapt_layer') and self.adapt_layer is not None:
                adapt_features = self.adapt_layer(features)
                return output, adapt_features
            else:
                # 如果没有适应层，返回原始特征
                return output, features
        
        return output
