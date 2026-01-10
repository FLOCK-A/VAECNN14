"""
Knowledge Distillation Loss Functions

Implements KD loss with temperature scaling, weighting strategies, and scheduling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class KDLossConfig:
    """Configuration for KD loss computation"""
    temperature: float = 4.0
    kd_alpha: float = 0.7  # Weight for KD loss (1-alpha for CE loss)
    loss_type: str = 'kldiv'  # 'kldiv' or 'soft_ce'
    kd_weighting: str = 'none'  # 'none', 'pmax', 'threshold'
    confidence_threshold: float = 0.0  # For threshold weighting
    warmup_epochs: int = 0  # Epochs with only CE loss
    kd_ramp_epochs: int = 0  # Epochs to linearly ramp up KD weight


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    loss_type: str = 'kldiv',
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    Compute knowledge distillation loss between student and teacher logits.
    
    Args:
        student_logits: Student model logits [B, num_classes]
        teacher_logits: Teacher model logits [B, num_classes] (will be detached)
        temperature: Temperature for softening probability distributions
        loss_type: 'kldiv' (KL divergence) or 'soft_ce' (soft cross-entropy)
        reduction: 'batchmean', 'mean', 'sum', or 'none'
        
    Returns:
        KD loss (scalar if reduction != 'none', otherwise [B])
        
    Notes:
        - Teacher logits are automatically detached
        - For KLDiv: Uses F.kl_div with log_target=False
        - For soft_ce: Uses cross-entropy with soft labels
        - Temperature scaling applied before computing loss
        - Loss is scaled by T^2 to maintain gradient magnitude
    """
    # Detach teacher to prevent gradients
    teacher_logits = teacher_logits.detach()
    
    # Apply temperature scaling
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    if loss_type == 'kldiv':
        # KL divergence: KL(teacher || student)
        # F.kl_div expects log-probabilities as input, probabilities as target
        loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction=reduction,
            log_target=False
        )
    elif loss_type == 'soft_ce':
        # Soft cross-entropy: -sum(teacher_soft * log(student_soft))
        loss = -(teacher_soft * student_soft).sum(dim=1)
        if reduction == 'batchmean' or reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # else: reduction == 'none', keep per-sample
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'kldiv' or 'soft_ce'")
    
    # Scale by temperature^2 to maintain gradient magnitude
    # This compensates for the 1/T^2 effect of temperature scaling on gradients
    loss = loss * (temperature ** 2)
    
    return loss


def compute_confidence_weights(
    teacher_logits: torch.Tensor,
    weighting: str = 'none',
    threshold: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute per-sample weights based on teacher confidence.
    
    Args:
        teacher_logits: Teacher logits [B, num_classes]
        weighting: Weighting strategy
            - 'none': All weights = 1.0
            - 'pmax': Weight by max probability (teacher confidence)
            - 'threshold': Binary mask (1 if pmax > threshold, 0 otherwise)
        threshold: Confidence threshold for 'threshold' weighting
        
    Returns:
        weights: Per-sample weights [B]
        stats: Dict with coverage rate and average confidence
    """
    with torch.no_grad():
        # Get teacher probabilities
        teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
        pmax, _ = teacher_probs.max(dim=1)  # [B]
        
        # Compute weights
        if weighting == 'none':
            weights = torch.ones_like(pmax)
            coverage = 1.0
        elif weighting == 'pmax':
            weights = pmax
            coverage = 1.0  # All samples included but weighted
        elif weighting == 'threshold':
            weights = (pmax > threshold).float()
            coverage = weights.mean().item()
        else:
            raise ValueError(f"Unknown weighting: {weighting}. Use 'none', 'pmax', or 'threshold'")
        
        # Compute statistics
        avg_confidence = pmax.mean().item()
        
        stats = {
            'coverage': coverage,
            'avg_confidence': avg_confidence,
            'min_confidence': pmax.min().item(),
            'max_confidence': pmax.max().item()
        }
    
    return weights, stats


def compute_kd_weight(
    current_epoch: int,
    warmup_epochs: int = 0,
    kd_ramp_epochs: int = 0,
    target_alpha: float = 0.7
) -> float:
    """
    Compute KD loss weight based on training schedule.
    
    Args:
        current_epoch: Current training epoch (0-indexed)
        warmup_epochs: Number of warmup epochs (only CE, no KD)
        kd_ramp_epochs: Number of epochs to linearly ramp up KD weight
        target_alpha: Target KD weight after warmup + ramp
        
    Returns:
        kd_alpha: Weight for KD loss (0.0 to target_alpha)
        
    Schedule:
        - Epochs 0 to warmup_epochs-1: kd_alpha = 0.0 (only CE)
        - Epochs warmup_epochs to warmup_epochs+kd_ramp_epochs-1: 
            kd_alpha linearly increases from 0.0 to target_alpha
        - Epochs >= warmup_epochs+kd_ramp_epochs: kd_alpha = target_alpha
    """
    if current_epoch < warmup_epochs:
        # Warmup: only CE loss
        return 0.0
    elif current_epoch < warmup_epochs + kd_ramp_epochs:
        # Ramp: linearly increase KD weight
        ramp_progress = (current_epoch - warmup_epochs) / kd_ramp_epochs
        return target_alpha * ramp_progress
    else:
        # Full KD weight
        return target_alpha


def compute_student_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    config: KDLossConfig,
    current_epoch: int = 0,
    domain_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined student loss (CE + KD) with scheduling and weighting.
    
    Args:
        student_logits: Student logits [B, num_classes]
        teacher_logits: Teacher logits [B, num_classes]
        targets: Ground truth labels [B]
        config: KDLossConfig with all hyperparameters
        current_epoch: Current training epoch (for scheduling)
        domain_mask: Optional boolean mask [B] for target domain samples
            If provided, KD weighting only applied to True samples
            
    Returns:
        loss: Combined scalar loss
        stats: Dict with detailed loss breakdown and statistics
            - 'total_loss': Combined loss value
            - 'ce_loss': Cross-entropy loss value
            - 'kd_loss': KD loss value (0 if in warmup)
            - 'kd_weight': Current KD weight (alpha)
            - 'ce_weight': Current CE weight (1-alpha)
            - 'coverage': Fraction of samples with KD applied
            - 'avg_confidence': Average teacher confidence
    """
    batch_size = student_logits.size(0)
    
    # Compute KD weight based on schedule
    kd_alpha = compute_kd_weight(
        current_epoch,
        config.warmup_epochs,
        config.kd_ramp_epochs,
        config.kd_alpha
    )
    ce_alpha = 1.0 - kd_alpha
    
    # Compute CE loss (always computed)
    ce_loss = F.cross_entropy(student_logits, targets, reduction='mean')
    
    # Initialize stats
    stats = {
        'total_loss': ce_loss.item(),
        'ce_loss': ce_loss.item(),
        'kd_loss': 0.0,
        'kd_weight': kd_alpha,
        'ce_weight': ce_alpha,
        'coverage': 0.0,
        'avg_confidence': 0.0
    }
    
    # Compute KD loss if not in warmup
    if kd_alpha > 0:
        # Compute confidence weights
        weights, weight_stats = compute_confidence_weights(
            teacher_logits,
            config.kd_weighting,
            config.confidence_threshold
        )
        
        # Apply domain mask if provided
        if domain_mask is not None:
            weights = weights * domain_mask.float()
            effective_coverage = (weights > 0).float().mean().item()
        else:
            effective_coverage = weight_stats['coverage']
        
        # Compute per-sample KD loss
        kd_loss_per_sample = kd_loss(
            student_logits,
            teacher_logits,
            temperature=config.temperature,
            loss_type=config.loss_type,
            reduction='none'
        )
        
        # Apply weights and reduce
        weighted_kd_loss = (kd_loss_per_sample * weights).sum() / max(weights.sum(), 1e-8)
        
        # Combine losses
        total_loss = ce_alpha * ce_loss + kd_alpha * weighted_kd_loss
        
        # Update stats
        stats.update({
            'total_loss': total_loss.item(),
            'kd_loss': weighted_kd_loss.item(),
            'coverage': effective_coverage,
            'avg_confidence': weight_stats['avg_confidence']
        })
        
        return total_loss, stats
    else:
        # Warmup: only CE loss
        return ce_loss, stats


# Convenience function for backward compatibility
def compute_kd_loss_with_schedule(*args, **kwargs):
    """Alias for compute_student_loss for backward compatibility"""
    return compute_student_loss(*args, **kwargs)
