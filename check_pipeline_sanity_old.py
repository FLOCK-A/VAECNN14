#!/usr/bin/env python3
"""
Pipeline Sanity Check

Tests complete pipeline (teacher → cache → student) with minimal data
to ensure all components work correctly without crashing.

Usage:
    python check_pipeline_sanity.py --json_path data/sample_dataset_extended.json --data_root data/
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a command and check for success"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False


def check_pipeline_sanity(json_path, data_root, num_scenes=10, num_cities=12):
    """
    Run complete pipeline sanity check with minimal data.
    
    Tests:
    1. Train City Teacher (1 epoch)
    2. Train City2Scene Teacher (1 epoch)
    3. Train Scene Teacher (1 epoch)
    4. Cache teacher logits
    5. Train Student (ce_only, 1 epoch)
    6. Train Student (city2scene, 1 epoch)
    
    Args:
        json_path: Path to minimal dataset JSON
        data_root: Root directory for data
        num_scenes: Number of scene classes
        num_cities: Number of city classes
    
    Returns:
        bool: True if all steps pass, False otherwise
    """
    target_name = 'b'  # Use target 'b' for testing
    batch_size = 4
    epochs = 1
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix='sanity_check_')
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # Step 1: Train City Teacher
        city_teacher_ckpt = os.path.join(temp_dir, 'teacher_city', f'A2{target_name}', 'best.pth')
        cmd = [
            'python', 'train_teacher_city.py',
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', target_name,
            '--num_scenes', str(num_scenes),
            '--num_cities', str(num_cities),
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--output_dir', temp_dir
        ]
        if not run_command(cmd, "Step 1: Train City Teacher"):
            return False
        
        if not os.path.exists(city_teacher_ckpt):
            print(f"✗ City Teacher checkpoint not found: {city_teacher_ckpt}")
            return False
        
        # Step 2: Train City2Scene Teacher
        city2scene_ckpt = os.path.join(temp_dir, 'teacher_city2scene', f'A2{target_name}', 'best.pth')
        cmd = [
            'python', 'train_teacher_city2scene.py',
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', target_name,
            '--city_teacher_ckpt', city_teacher_ckpt,
            '--num_scenes', str(num_scenes),
            '--num_cities', str(num_cities),
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--output_dir', temp_dir
        ]
        if not run_command(cmd, "Step 2: Train City2Scene Teacher"):
            return False
        
        if not os.path.exists(city2scene_ckpt):
            print(f"✗ City2Scene Teacher checkpoint not found: {city2scene_ckpt}")
            return False
        
        # Step 3: Train Scene Teacher
        scene_ckpt = os.path.join(temp_dir, 'teacher_scene', 'global', 'best.pth')
        cmd = [
            'python', 'train_teacher_scene.py',
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', 'global',
            '--num_scenes', str(num_scenes),
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--output_dir', temp_dir
        ]
        if not run_command(cmd, "Step 3: Train Scene Teacher"):
            return False
        
        if not os.path.exists(scene_ckpt):
            print(f"✗ Scene Teacher checkpoint not found: {scene_ckpt}")
            return False
        
        # Step 4: Cache teacher logits
        cache_root = os.path.join(temp_dir, 'cache')
        cache_index = os.path.join(cache_root, f'A2{target_name}', 'cache_index.json')
        cmd = [
            'python', 'dump_teacher_logits.py',
            '--city2scene_teacher_ckpt', city2scene_ckpt,
            '--scene_teacher_ckpt', scene_ckpt,
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', target_name,
            '--num_scenes', str(num_scenes),
            '--cache_root', cache_root,
            '--num_verify', '3'
        ]
        if not run_command(cmd, "Step 4: Cache Teacher Logits"):
            return False
        
        if not os.path.exists(cache_index):
            print(f"✗ Cache index not found: {cache_index}")
            return False
        
        # Step 5: Train Student (ce_only baseline)
        student_ce_ckpt = os.path.join(temp_dir, 'student', f'A2{target_name}', 'ce_only', 'best.pth')
        cmd = [
            'python', 'train_student_distill.py',
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', target_name,
            '--num_scenes', str(num_scenes),
            '--teacher_mode', 'ce_only',
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--output_dir', temp_dir
        ]
        if not run_command(cmd, "Step 5: Train Student (ce_only)"):
            return False
        
        if not os.path.exists(student_ce_ckpt):
            print(f"✗ Student (ce_only) checkpoint not found: {student_ce_ckpt}")
            return False
        
        # Step 6: Train Student (city2scene with KD)
        student_kd_ckpt = os.path.join(temp_dir, 'student', f'A2{target_name}', 'city2scene', 'best.pth')
        cmd = [
            'python', 'train_student_distill.py',
            '--json_path', json_path,
            '--data_root', data_root,
            '--target_name', target_name,
            '--num_scenes', str(num_scenes),
            '--teacher_mode', 'city2scene',
            '--cache_root', os.path.join(cache_root, f'A2{target_name}'),
            '--temperature', '4.0',
            '--kd_alpha', '0.7',
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--output_dir', temp_dir
        ]
        if not run_command(cmd, "Step 6: Train Student (city2scene with KD)"):
            return False
        
        if not os.path.exists(student_kd_ckpt):
            print(f"✗ Student (city2scene) checkpoint not found: {student_kd_ckpt}")
            return False
        
        print(f"\n{'='*80}")
        print("✓ ALL SANITY CHECKS PASSED!")
        print('='*80)
        print(f"\nTemporary outputs saved in: {temp_dir}")
        print("You can delete this directory when done inspecting.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline sanity check failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Pipeline Sanity Check')
    parser.add_argument('--json_path', type=str, required=True,
                      help='Path to minimal dataset JSON')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Root directory for data files')
    parser.add_argument('--num_scenes', type=int, default=10,
                      help='Number of scene classes (default: 10)')
    parser.add_argument('--num_cities', type=int, default=12,
                      help='Number of city classes (default: 12)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found: {args.data_root}")
        sys.exit(1)
    
    # Run sanity check
    success = check_pipeline_sanity(
        args.json_path,
        args.data_root,
        args.num_scenes,
        args.num_cities
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
