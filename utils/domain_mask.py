"""
Domain mask utilities for replacing hardcoded batch splitting logic.
"""
import torch


def create_domain_mask(batch, source_domain=0):
    """
    Create boolean masks for source/target domains.
    Replaces "first half/second half" batch splitting assumptions.
    
    Args:
        batch: dict with 'domain' field (torch.Tensor of shape [B])
        source_domain: int - ID of source domain (default=0 for device A)
    
    Returns:
        source_mask: torch.BoolTensor [B] - True for source samples
        target_mask: torch.BoolTensor [B] - True for target samples
    """
    if 'domain' not in batch:
        raise KeyError("Batch must contain 'domain' field for domain masking")
    
    domain = batch['domain']
    source_mask = (domain == source_domain)
    target_mask = ~source_mask
    
    return source_mask, target_mask


def split_by_domain(batch, source_domain=0):
    """
    Split batch into source and target subsets based on domain.
    
    Args:
        batch: dict with tensor fields
        source_domain: int - ID of source domain
    
    Returns:
        source_batch: dict - subset with source domain samples
        target_batch: dict - subset with target domain samples
    """
    source_mask, target_mask = create_domain_mask(batch, source_domain)
    
    source_batch = {}
    target_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            source_batch[key] = value[source_mask]
            target_batch[key] = value[target_mask]
        elif isinstance(value, list):
            # For lists (like 'split', 'device_id', 'path')
            source_batch[key] = [v for i, v in enumerate(value) if source_mask[i]]
            target_batch[key] = [v for i, v in enumerate(value) if target_mask[i]]
        else:
            # Keep non-indexable items as-is
            source_batch[key] = value
            target_batch[key] = value
    
    return source_batch, target_batch


def get_source_samples(batch, source_domain=0):
    """
    Extract only source domain samples from batch.
    
    Args:
        batch: dict with tensor fields
        source_domain: int - ID of source domain
        
    Returns:
        source_batch: dict - subset with source domain samples only
    """
    source_mask, _ = create_domain_mask(batch, source_domain)
    
    source_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            source_batch[key] = value[source_mask]
        elif isinstance(value, list):
            source_batch[key] = [v for i, v in enumerate(value) if source_mask[i]]
        else:
            source_batch[key] = value
    
    return source_batch


def get_target_samples(batch, source_domain=0):
    """
    Extract only target domain samples from batch.
    
    Args:
        batch: dict with tensor fields
        source_domain: int - ID of source domain
        
    Returns:
        target_batch: dict - subset with target domain samples only
    """
    _, target_mask = create_domain_mask(batch, source_domain)
    
    target_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            target_batch[key] = value[target_mask]
        elif isinstance(value, list):
            target_batch[key] = [v for i, v in enumerate(value) if target_mask[i]]
        else:
            target_batch[key] = value
    
    return target_batch
