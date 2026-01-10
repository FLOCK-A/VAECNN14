"""
Data validation utilities for enforcing experimental protocol.
Ensures no data leakage from test splits or unseen devices during training.
"""

def validate_no_leakage(samples, allowed_splits=None, allowed_devices=None, phase='training', verbose=True):
    """
    Validate that samples don't violate experimental protocol.
    
    Enhanced with detailed violation reporting including sample paths.
    
    Args:
        samples: List of sample dicts
        allowed_splits: List of allowed split values (e.g., ['train']), None means no check
        allowed_devices: List of allowed device IDs (e.g., [0, 1]), None means no check
        phase: str - description of current phase for error message
        verbose: bool - if True, print detailed violation paths
        
    Raises:
        ValueError: If any sample violates the constraints with detailed paths
    """
    if allowed_splits is None and allowed_devices is None:
        return  # No validation needed
    
    violations = []
    violation_paths = []
    
    for idx, sample in enumerate(samples):
        sample_id = sample.get('file', f'index_{idx}')
        sample_path = sample.get('path', sample.get('file', f'unknown_path_{idx}'))
        
        # Check split constraint
        if allowed_splits is not None:
            split = sample.get('split', 'train')  # default to 'train' for backward compatibility
            if split not in allowed_splits:
                violations.append(
                    f"Sample {sample_id}: split='{split}' not allowed in {phase} "
                    f"(allowed: {allowed_splits})"
                )
                violation_paths.append(f"  PATH: {sample_path} [split={split}]")
        
        # Check device constraint (unseen devices)
        if allowed_devices is not None:
            domain = sample.get('domain')
            if domain not in allowed_devices:
                device_id = sample.get('device_id', f'domain_{domain}')
                violations.append(
                    f"Sample {sample_id}: device={device_id} (domain={domain}) not allowed in {phase} "
                    f"(allowed domains: {allowed_devices})"
                )
                violation_paths.append(f"  PATH: {sample_path} [device={device_id}, domain={domain}]")
    
    if violations:
        error_msg = f"\n{'='*80}\n🚨 DATA LEAKAGE DETECTED in {phase}! 🚨\n{'='*80}\n"
        error_msg += f"Found {len(violations)} violation(s):\n\n"
        
        # Show first 20 violations with paths
        max_show = 20
        for i, (viol, path) in enumerate(zip(violations[:max_show], violation_paths[:max_show])):
            error_msg += f"{i+1}. {viol}\n{path}\n\n"
        
        if len(violations) > max_show:
            error_msg += f"... and {len(violations) - max_show} more violations\n\n"
        
        error_msg += f"{'='*80}\n"
        error_msg += "⚠️  CRITICAL: Training data contains forbidden samples!\n"
        error_msg += "Please ensure:\n"
        error_msg += "  - split='test' samples are excluded from training\n"
        error_msg += "  - Unseen devices (s4, s5, s6 / domains 6-8) are excluded from training\n"
        error_msg += f"{'='*80}\n"
        raise ValueError(error_msg)


def validate_label_availability(samples, domain_id, label_type, phase='training'):
    """
    Validate that required labels are available for the specified domain.
    
    Args:
        samples: List of sample dicts
        domain_id: Domain ID to check (e.g., 0 for source, 1 for target)
        label_type: 'scene' or 'city' - which label type is required
        phase: str - description of current phase for error message
        
    Raises:
        ValueError: If required labels are missing
    """
    label_field = f'{label_type}_label'
    missing = []
    
    for idx, sample in enumerate(samples):
        if sample.get('domain') == domain_id:
            if label_field not in sample or sample[label_field] is None:
                sample_id = sample.get('file', f'index_{idx}')
                missing.append(f"Sample {sample_id}: missing '{label_field}'")
    
    if missing:
        error_msg = f"\n{'='*80}\nMISSING LABELS in {phase}!\n{'='*80}\n"
        error_msg += f"Domain {domain_id} requires '{label_type}' labels but found missing values:\n"
        error_msg += "\n".join(missing[:10])  # Show first 10
        if len(missing) > 10:
            error_msg += f"\n... and {len(missing) - 10} more"
        error_msg += f"\n{'='*80}\n"
        raise ValueError(error_msg)


def verify_backbone_frozen(model, model_name='model'):
    """
    Verify that backbone parameters are frozen (requires_grad=False).
    
    Args:
        model: PyTorch model with feature_extractor attribute
        model_name: Name of the model for error messages
        
    Raises:
        RuntimeError: If any backbone parameters have requires_grad=True
    """
    if not hasattr(model, 'feature_extractor'):
        print(f"⚠️  Warning: {model_name} has no 'feature_extractor' attribute - skipping freeze check")
        return
    
    unfrozen_params = []
    total_backbone_params = 0
    
    for name, param in model.feature_extractor.named_parameters():
        total_backbone_params += 1
        if param.requires_grad:
            unfrozen_params.append(name)
    
    if unfrozen_params:
        error_msg = f"\n{'='*80}\n"
        error_msg += f"🚨 BACKBONE NOT FULLY FROZEN in {model_name}! 🚨\n"
        error_msg += f"{'='*80}\n"
        error_msg += f"Found {len(unfrozen_params)}/{total_backbone_params} backbone parameters with requires_grad=True:\n\n"
        
        # Show first 10
        for i, pname in enumerate(unfrozen_params[:10]):
            error_msg += f"  {i+1}. {pname}\n"
        
        if len(unfrozen_params) > 10:
            error_msg += f"  ... and {len(unfrozen_params) - 10} more\n"
        
        error_msg += f"\n{'='*80}\n"
        error_msg += "⚠️  CRITICAL: Backbone should be completely frozen for Stage-2 training!\n"
        error_msg += "Please ensure all feature_extractor parameters have requires_grad=False.\n"
        error_msg += f"{'='*80}\n"
        raise RuntimeError(error_msg)
    
    print(f"✓ Backbone freeze verification passed: {total_backbone_params} parameters frozen")


def verify_cache_exists(cache_root, file_paths, teacher_mode, raise_on_missing=True):
    """
    Verify that cached teacher logits exist for all required samples.
    
    Args:
        cache_root: Root directory of cache (e.g., cache/A2b)
        file_paths: List of file paths to check
        teacher_mode: Teacher type ('city2scene' or 'scene')
        raise_on_missing: If True, raise error on missing cache; if False, return list
        
    Returns:
        List of missing paths (if raise_on_missing=False)
        
    Raises:
        FileNotFoundError: If cache files are missing and raise_on_missing=True
    """
    import os
    import json
    import hashlib
    
    # Load cache index
    index_path = os.path.join(cache_root, 'cache_index.json')
    if not os.path.exists(index_path):
        if raise_on_missing:
            error_msg = f"\n{'='*80}\n"
            error_msg += f"🚨 CACHE INDEX NOT FOUND! 🚨\n"
            error_msg += f"{'='*80}\n"
            error_msg += f"Expected: {index_path}\n\n"
            error_msg += "Please run dump_teacher_logits.py to cache teacher logits before student training.\n"
            error_msg += f"{'='*80}\n"
            raise FileNotFoundError(error_msg)
        else:
            return file_paths
    
    with open(index_path, 'r') as f:
        cache_index = json.load(f)
    
    if teacher_mode not in cache_index:
        if raise_on_missing:
            error_msg = f"\n{'='*80}\n"
            error_msg += f"🚨 TEACHER MODE NOT IN CACHE! 🚨\n"
            error_msg += f"{'='*80}\n"
            error_msg += f"Teacher mode '{teacher_mode}' not found in cache index.\n"
            error_msg += f"Available: {list(cache_index.keys())}\n\n"
            error_msg += "Please run dump_teacher_logits.py with the correct teacher checkpoints.\n"
            error_msg += f"{'='*80}\n"
            raise FileNotFoundError(error_msg)
        else:
            return file_paths
    
    teacher_cache = cache_index[teacher_mode]
    missing_paths = []
    
    for fpath in file_paths:
        # Extract just the filename
        fname = os.path.basename(fpath)
        if fname not in teacher_cache:
            missing_paths.append(fpath)
    
    if missing_paths and raise_on_missing:
        error_msg = f"\n{'='*80}\n"
        error_msg += f"🚨 CACHE MISSING FOR {len(missing_paths)} SAMPLES! 🚨\n"
        error_msg += f"{'='*80}\n"
        error_msg += f"Teacher mode: {teacher_mode}\n"
        error_msg += f"Cache root: {cache_root}\n\n"
        error_msg += "Missing cache for the following samples:\n\n"
        
        for i, path in enumerate(missing_paths[:20]):
            error_msg += f"  {i+1}. {path}\n"
        
        if len(missing_paths) > 20:
            error_msg += f"  ... and {len(missing_paths) - 20} more\n"
        
        error_msg += f"\n{'='*80}\n"
        error_msg += "⚠️  Please run dump_teacher_logits.py to cache all required samples.\n"
        error_msg += f"{'='*80}\n"
        raise FileNotFoundError(error_msg)
    
    if not raise_on_missing:
        return missing_paths
    
    print(f"✓ Cache verification passed: {len(file_paths)} samples found for {teacher_mode}")

