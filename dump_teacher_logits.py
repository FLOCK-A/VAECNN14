#!/usr/bin/env python3
"""
Teacher Logits Offline Caching Script

Dumps teacher model logits to disk for faster student training.
Supports incremental caching and reproducibility.

Output structure:
  cache/A2{target}/city2scene/{hash}.npy
  cache/A2{target}/scene/{hash}.npy
  cache/A2{target}/cache_index.json
"""
import os
import sys
import json
import argparse
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classifier import Cnn14Classifier
from data.dataloader import get_dataloader
from utils.data_validation import validate_no_leakage
from utils.dataset_io import load_dataset


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_to_hash(file_path):
    """Convert file path to hash for cache filename"""
    return hashlib.md5(file_path.encode('utf-8')).hexdigest()


def load_teacher_model(checkpoint_path, num_classes, device, model_type='full'):
    """
    Load a teacher model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        num_classes: Number of output classes
        device: torch.device
        model_type: 'full' for full model, 'city2scene' for city+scene_head
        
    Returns:
        model: Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'city2scene':
        # Load city model + scene head
        city_model = Cnn14Classifier(
            classes_num=num_classes,  # Will be overridden by checkpoint
            checkpoint_path=None,
            use_projection_head=False,
            dropout=0.3,
            use_adapt_head=False
        )
        
        # Load city model state
        city_model.load_state_dict(checkpoint['city_model_state_dict'])
        city_model = city_model.to(device)
        city_model.eval()
        
        # Load scene head
        # Get feature dim from checkpoint or use default
        feature_dim = 2048  # Default
        num_scenes = 10  # Will be inferred from scene_head
        
        # Infer num_scenes from scene_head weight shape
        scene_head_weight = checkpoint['scene_head_state_dict']['weight']
        num_scenes = scene_head_weight.shape[0]
        feature_dim = scene_head_weight.shape[1]
        
        scene_head = nn.Linear(feature_dim, num_scenes)
        scene_head.load_state_dict(checkpoint['scene_head_state_dict'])
        scene_head = scene_head.to(device)
        scene_head.eval()
        
        # Create combined model
        class City2SceneTeacher(nn.Module):
            def __init__(self, city_model, scene_head):
                super().__init__()
                self.city_model = city_model
                self.scene_head = scene_head
            
            def forward(self, x):
                with torch.no_grad():
                    features = self.city_model.forward_features(x)
                    logits = self.scene_head(features)
                return logits
        
        model = City2SceneTeacher(city_model, scene_head)
        
    else:
        # Load full model (scene teacher or city teacher)
        model = Cnn14Classifier(
            classes_num=num_classes,
            checkpoint_path=None,
            use_projection_head=False,
            dropout=0.3,
            use_adapt_head=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    model.eval()
    return model


def dump_logits(teacher_model, samples, data_root, cache_dir, teacher_name,
                use_fp16=False, skip_existing=True, device='cuda'):
    """
    Dump teacher logits for all samples
    
    Args:
        teacher_model: Teacher model
        samples: List of sample dicts
        data_root: Root directory for NPY features
        cache_dir: Cache directory for this teacher
        teacher_name: Name of teacher ('city2scene' or 'scene')
        use_fp16: Save as fp16
        skip_existing: Skip if cache file exists
        device: Device to run on
        
    Returns:
        cache_map: Dict mapping file paths to cache files
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_map = {}
    skipped = 0
    cached = 0
    
    print(f"\nDumping {teacher_name} teacher logits to: {cache_dir}")
    print(f"Total samples: {len(samples)}")
    
    teacher_model.eval()
    
    for sample in tqdm(samples, desc=f"Caching {teacher_name}"):
        file_path = sample.get('file', '')
        
        # Generate cache filename
        file_hash = path_to_hash(file_path)
        cache_file = os.path.join(cache_dir, f"{file_hash}.npy")
        
        # Skip if exists
        if skip_existing and os.path.exists(cache_file):
            cache_map[file_path] = cache_file
            skipped += 1
            continue
        
        # Load features
        npy_path = os.path.join(data_root, file_path) if not os.path.isabs(file_path) else file_path
        features = np.load(npy_path)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = teacher_model(features_tensor)
            logits_np = logits.cpu().numpy().squeeze(0)
            
            # Convert to fp16 if requested
            if use_fp16:
                logits_np = logits_np.astype(np.float16)
        
        # Save to cache
        np.save(cache_file, logits_np)
        cache_map[file_path] = cache_file
        cached += 1
    
    print(f"  Cached: {cached}, Skipped (existing): {skipped}")
    
    return cache_map


def load_and_validate_data(json_path, target_name, source_domain=0):
    """
    Load dataset and validate for caching
    
    Args:
        json_path: Path to dataset JSON
        target_name: Target device name
        source_domain: Source domain ID
        
    Returns:
        samples: List of samples to cache (A_train + target_train)
    """
    # Use unified dataset loader
    dataset = load_dataset(json_path)
    
    # Get training samples (split=train only)
    train_split = dataset['train']
    
    # Map target name to domain ID
    target_domain_map = {'b': 1, 'c': 2, 's1': 3, 's2': 4, 's3': 5}
    if target_name not in target_domain_map:
        raise ValueError(f"Invalid target_name: {target_name}. Must be one of {list(target_domain_map.keys())}")
    target_domain = target_domain_map[target_name]
    
    # Filter samples: A_train + target_train
    source_samples = [s for s in train_split if s.get('domain') == source_domain]
    target_samples = [s for s in train_split if s.get('domain') == target_domain]
    
    # Validate no data leakage
    all_samples = source_samples + target_samples
    validate_no_leakage(all_samples,
                       allowed_splits=['train'],
                       allowed_devices=[source_domain, target_domain],
                       phase='Teacher logits caching')
    
    print(f"Source domain (A) samples: {len(source_samples)}")
    print(f"Target domain ({target_name}) samples: {len(target_samples)}")
    print(f"Total samples to cache: {len(all_samples)}")
    
    return all_samples


def save_cache_index(cache_root, city2scene_map, scene_map):
    """
    Save cache index JSON for fast lookup
    
    Args:
        cache_root: Root cache directory
        city2scene_map: Path->cache mapping for city2scene teacher
        scene_map: Path->cache mapping for scene teacher
    """
    cache_index = {
        'city2scene': city2scene_map,
        'scene': scene_map,
        'num_samples': len(city2scene_map)
    }
    
    index_path = os.path.join(cache_root, 'cache_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(cache_index, f, indent=2, ensure_ascii=False)
    
    print(f"\nCache index saved to: {index_path}")
    print(f"Total cached samples: {cache_index['num_samples']}")


def verify_cache(cache_root, samples, num_verify=10):
    """
    Verify cache by randomly reading samples
    
    Args:
        cache_root: Root cache directory
        samples: List of all samples
        num_verify: Number of samples to verify
    """
    print(f"\n{'='*80}")
    print("Cache Verification")
    print(f"{'='*80}")
    
    # Load cache index
    index_path = os.path.join(cache_root, 'cache_index.json')
    with open(index_path, 'r', encoding='utf-8') as f:
        cache_index = json.load(f)
    
    # Randomly select samples to verify
    random.seed(42)
    verify_samples = random.sample(samples, min(num_verify, len(samples)))
    
    city2scene_hits = 0
    scene_hits = 0
    
    print(f"Verifying {len(verify_samples)} random samples...")
    
    for sample in verify_samples:
        file_path = sample.get('file', '')
        
        # Check city2scene cache
        if file_path in cache_index['city2scene']:
            cache_file = cache_index['city2scene'][file_path]
            if os.path.exists(cache_file):
                logits = np.load(cache_file)
                city2scene_hits += 1
        
        # Check scene cache
        if file_path in cache_index['scene']:
            cache_file = cache_index['scene'][file_path]
            if os.path.exists(cache_file):
                logits = np.load(cache_file)
                scene_hits += 1
    
    city2scene_hit_rate = 100.0 * city2scene_hits / len(verify_samples)
    scene_hit_rate = 100.0 * scene_hits / len(verify_samples)
    
    print(f"\nCity2Scene Teacher:")
    print(f"  Hits: {city2scene_hits}/{len(verify_samples)}")
    print(f"  Hit Rate: {city2scene_hit_rate:.1f}%")
    
    print(f"\nScene Teacher:")
    print(f"  Hits: {scene_hits}/{len(verify_samples)}")
    print(f"  Hit Rate: {scene_hit_rate:.1f}%")
    
    if city2scene_hit_rate == 100.0 and scene_hit_rate == 100.0:
        print(f"\n✓ Verification PASSED: 100% hit rate for both teachers")
    else:
        print(f"\n✗ Verification FAILED: Hit rate < 100%")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Dump teacher logits to cache')
    parser.add_argument('--city2scene_teacher_ckpt', type=str, required=True,
                       help='Path to City2Scene teacher checkpoint')
    parser.add_argument('--scene_teacher_ckpt', type=str, required=True,
                       help='Path to Scene teacher checkpoint')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to dataset JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for NPY features')
    parser.add_argument('--target_name', type=str, required=True,
                       choices=['b', 'c', 's1', 's2', 's3'],
                       help='Target device name')
    parser.add_argument('--cache_root', type=str, default='cache',
                       help='Root cache directory (default: cache)')
    parser.add_argument('--num_scenes', type=int, default=10,
                       help='Number of scene classes')
    parser.add_argument('--num_cities', type=int, default=12,
                       help='Number of city classes')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Save logits in fp16 format')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='Skip caching if file exists (default: True)')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                       help='Force re-cache all files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_verify', type=int, default=10,
                       help='Number of samples to verify (default: 10)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and validate data
    print("\n" + "="*80)
    print("Loading and validating data...")
    print("="*80)
    samples = load_and_validate_data(args.json_path, args.target_name)
    
    # Create cache directories
    cache_root = os.path.join(args.cache_root, f'A2{args.target_name}')
    city2scene_cache_dir = os.path.join(cache_root, 'city2scene')
    scene_cache_dir = os.path.join(cache_root, 'scene')
    
    # Load teachers
    print("\n" + "="*80)
    print("Loading teacher models...")
    print("="*80)
    
    print(f"Loading City2Scene teacher from: {args.city2scene_teacher_ckpt}")
    city2scene_teacher = load_teacher_model(
        args.city2scene_teacher_ckpt,
        args.num_scenes,
        device,
        model_type='city2scene'
    )
    print("✓ City2Scene teacher loaded")
    
    print(f"\nLoading Scene teacher from: {args.scene_teacher_ckpt}")
    scene_teacher = load_teacher_model(
        args.scene_teacher_ckpt,
        args.num_scenes,
        device,
        model_type='full'
    )
    print("✓ Scene teacher loaded")
    
    # Dump logits
    print("\n" + "="*80)
    print("Dumping teacher logits...")
    print("="*80)
    
    city2scene_map = dump_logits(
        city2scene_teacher,
        samples,
        args.data_root,
        city2scene_cache_dir,
        'city2scene',
        use_fp16=args.use_fp16,
        skip_existing=args.skip_existing,
        device=device
    )
    
    scene_map = dump_logits(
        scene_teacher,
        samples,
        args.data_root,
        scene_cache_dir,
        'scene',
        use_fp16=args.use_fp16,
        skip_existing=args.skip_existing,
        device=device
    )
    
    # Save cache index
    save_cache_index(cache_root, city2scene_map, scene_map)
    
    # Verify cache
    verify_result = verify_cache(cache_root, samples, args.num_verify)
    
    print("\n" + "="*80)
    print("Cache dumping completed!")
    print("="*80)
    print(f"Cache directory: {cache_root}")
    print(f"  - city2scene/: {len(city2scene_map)} files")
    print(f"  - scene/: {len(scene_map)} files")
    print(f"  - cache_index.json")
    print(f"\nFormat: {'fp16' if args.use_fp16 else 'fp32'}")
    print(f"Verification: {'✓ PASSED' if verify_result else '✗ FAILED'}")


if __name__ == '__main__':
    main()
