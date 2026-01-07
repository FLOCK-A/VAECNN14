"""
基于Cnn14LogMel的分类器模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        
        # ====== Disentangle heads ======
        feature_dim = config.FEATURE_DIM
        zs_dim = getattr(config, "ZS_DIM", 128)
        zd_dim = getattr(config, "ZD_DIM", 128)
        num_devices = getattr(config, "NUM_DEVICES", 6)

        # 分类头 - 基于特征维度进行分类
        self.dropout = nn.Dropout(dropout)
        
        # P_s: semantic projection (f -> z_s)
        self.P_s = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, zs_dim),
            nn.BatchNorm1d(zs_dim),
            nn.ReLU(inplace=True),
        )

        # P_d: device projection (f -> z_d)
        self.P_d = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, zd_dim),
            nn.BatchNorm1d(zd_dim),
            nn.ReLU(inplace=True),
        )

        # Scene classifier: C_s(z_s) -> scene logits
        self.scene_classifier = nn.Sequential(
            nn.Linear(zs_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, classes_num)
        )

        # Device classifier: C_d(z_d) -> device logits
        self.device_classifier = nn.Sequential(
            nn.Linear(zd_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_devices)
        )

        # 继续保留你原先用于CMD的低维特征（更稳）
        self.use_adapt_head = use_adapt_head and (adapt_dim is not None) and (adapt_dim > 0)
        if self.use_adapt_head:
            self.adapt_layer = nn.Sequential(
                nn.Linear(zs_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, adapt_dim),
                nn.BatchNorm1d(adapt_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.adapt_layer = None

    def forward(self, input_data, return_adapt_features=False, return_device_logits=False, return_latents=False):
        # backbone features
        f = self.feature_extractor(input_data)
        f = f.view(f.size(0), -1)
        f = self.dropout(f)

        # projections
        z_s = self.P_s(f)
        z_d = self.P_d(f)

        # heads
        scene_logits = self.scene_classifier(z_s)

        dev_logits = None
        if return_device_logits:
            dev_logits = self.device_classifier(z_d)

        # adapt features (for CMD)
        if return_adapt_features:
            if self.adapt_layer is not None:
                adapt_features = self.adapt_layer(z_s)
            else:
                adapt_features = z_s

            if return_device_logits and return_latents:
                return scene_logits, adapt_features, dev_logits, z_s, z_d
            if return_device_logits:
                return scene_logits, adapt_features, dev_logits
            if return_latents:
                return scene_logits, adapt_features, z_s, z_d
            return scene_logits, adapt_features

        # no adapt features
        if return_device_logits and return_latents:
            return scene_logits, dev_logits, z_s, z_d
        if return_device_logits:
            return scene_logits, dev_logits
        if return_latents:
            return scene_logits, z_s, z_d

        return scene_logits
