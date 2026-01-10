"""
Knowledge Distillation Loss Functions
"""

from .kd import kd_loss, compute_kd_weight, KDLossConfig

__all__ = ['kd_loss', 'compute_kd_weight', 'KDLossConfig']
