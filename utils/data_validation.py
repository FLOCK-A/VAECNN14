"""
Data validation utilities for enforcing experimental protocol.
Ensures no data leakage from test splits or unseen devices during training.
"""

def validate_no_leakage(samples, allowed_splits=None, allowed_devices=None, phase='training'):
    """
    Validate that samples don't violate experimental protocol.
    
    Args:
        samples: List of sample dicts
        allowed_splits: List of allowed split values (e.g., ['train']), None means no check
        allowed_devices: List of allowed device IDs (e.g., [0, 1]), None means no check
        phase: str - description of current phase for error message
        
    Raises:
        ValueError: If any sample violates the constraints
    """
    if allowed_splits is None and allowed_devices is None:
        return  # No validation needed
    
    violations = []
    
    for idx, sample in enumerate(samples):
        sample_id = sample.get('file', f'index_{idx}')
        
        # Check split constraint
        if allowed_splits is not None:
            split = sample.get('split', 'train')  # default to 'train' for backward compatibility
            if split not in allowed_splits:
                violations.append(
                    f"Sample {sample_id}: split='{split}' not allowed in {phase} "
                    f"(allowed: {allowed_splits})"
                )
        
        # Check device constraint
        if allowed_devices is not None:
            domain = sample.get('domain')
            if domain not in allowed_devices:
                device_id = sample.get('device_id', f'domain_{domain}')
                violations.append(
                    f"Sample {sample_id}: device={device_id} (domain={domain}) not allowed in {phase} "
                    f"(allowed domains: {allowed_devices})"
                )
    
    if violations:
        error_msg = f"\n{'='*80}\nDATA LEAKAGE DETECTED in {phase}!\n{'='*80}\n"
        error_msg += "\n".join(violations)
        error_msg += f"\n{'='*80}\n"
        error_msg += "Please ensure test split and unseen devices are excluded from training data.\n"
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
