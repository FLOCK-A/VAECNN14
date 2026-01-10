#!/usr/bin/env python3
"""
Enhanced Pipeline Sanity Check - CI/Local Gate

Comprehensive end-to-end pipeline testing with:
- Strict adherence to unified interfaces (dataset_io, path_key)
- Sequential step execution with skip capability
- Detailed logging and error reporting with command reproduction
- Dedicated output directory (outputs/_sanity/)
- Optional cleanup

Usage:
    python check_pipeline_sanity.py --json_path data/sample_dataset_extended.json --data_root data/
    
    # With cleanup
    python check_pipeline_sanity.py --json_path data/sample_dataset_extended.json --data_root data/ --cleanup
    
    # Skip certain steps
    python check_pipeline_sanity.py --json_path data/sample.json --data_root data/ --skip-step A,B
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import traceback
from pathlib import Path
from datetime import datetime


class PipelineSanityChecker:
    """
    Robust pipeline sanity checker for CI and local use.
    
    Tests complete pipeline with minimal data:
    - Step A: train_teacher_city
    - Step B: train_teacher_city2scene (with backbone freeze verification)
    - Step C: train_teacher_scene
    - Step D: dump_teacher_logits
    - Step E: verify_cache_exists (100% hit rate required)
    - Step F: train_student_distill (CE on source, KD on target)
    """
    
    STEP_NAMES = {
        'A': 'Train City Teacher',
        'B': 'Train City2Scene Teacher (with freeze check)',
        'C': 'Train Scene Teacher',
        'D': 'Dump Teacher Logits',
        'E': 'Verify Cache Existence (100% hit rate)',
        'F': 'Train Student (city2scene KD)'
    }
    
    def __init__(self, json_path, data_root, output_root='outputs/_sanity', 
                 num_scenes=10, num_cities=12, target_name='b',
                 batch_size=4, epochs=1, skip_steps=None):
        """
        Initialize pipeline checker.
        
        Args:
            json_path: Path to dataset JSON file
            data_root: Root directory for data files
            output_root: Root directory for outputs (default: outputs/_sanity)
            num_scenes: Number of scene classes
            num_cities: Number of city classes
            target_name: Target domain name
            batch_size: Batch size for training
            epochs: Number of epochs (typically 1 for sanity)
            skip_steps: List of step IDs to skip (e.g., ['A', 'B'])
        """
        self.json_path = os.path.abspath(json_path)
        self.data_root = os.path.abspath(data_root)
        self.output_root = os.path.abspath(output_root)
        self.num_scenes = num_scenes
        self.num_cities = num_cities
        self.target_name = target_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.skip_steps = skip_steps or []
        
        # Derived paths
        self.teacher_city_dir = os.path.join(self.output_root, 'teacher_city')
        self.teacher_city2scene_dir = os.path.join(self.output_root, 'teacher_city2scene')
        self.teacher_scene_dir = os.path.join(self.output_root, 'teacher_scene')
        self.cache_root = os.path.join(self.output_root, 'cache')
        self.student_dir = os.path.join(self.output_root, 'student')
        
        # Checkpoint paths
        self.city_teacher_ckpt = os.path.join(
            self.teacher_city_dir, f'A2{target_name}', 'best.pth')
        self.city2scene_ckpt = os.path.join(
            self.teacher_city2scene_dir, f'A2{target_name}', 'best.pth')
        self.scene_ckpt = os.path.join(
            self.teacher_scene_dir, 'global', 'best.pth')
        self.cache_index = os.path.join(
            self.cache_root, f'A2{target_name}', 'cache_index.json')
        self.student_ckpt = os.path.join(
            self.student_dir, f'A2{target_name}', 'city2scene', 'best.pth')
        
        # Results
        self.results = {}
        self.start_time = datetime.now()
    
    def log(self, message, level='INFO'):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        prefix = {
            'INFO': '→',
            'SUCCESS': '✓',
            'ERROR': '✗',
            'WARN': '⚠',
            'SKIP': '⊘'
        }.get(level, '•')
        print(f"[{timestamp}] {prefix} {message}")
    
    def print_separator(self, char='=', length=100):
        """Print separator line"""
        print(char * length)
    
    def run_command(self, cmd, step_id, description, expected_output=None):
        """
        Run a command with comprehensive logging and error handling.
        
        Args:
            cmd: Command list for subprocess
            step_id: Step identifier (A-F)
            description: Human-readable description
            expected_output: Path that should exist after successful execution
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.print_separator()
        self.log(f"Step {step_id}: {description}", 'INFO')
        self.print_separator()
        
        # Print full command for reproducibility
        cmd_str = ' '.join(cmd)
        self.log(f"Command: {cmd_str}", 'INFO')
        self.log(f"Working Directory: {os.getcwd()}", 'INFO')
        
        if expected_output:
            self.log(f"Expected Output: {expected_output}", 'INFO')
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # Log success
            self.log(f"Step {step_id} SUCCEEDED", 'SUCCESS')
            
            # Show last 20 lines of output for context
            stdout_lines = result.stdout.strip().split('\n')
            if len(stdout_lines) > 0:
                self.log(f"Output (last 20 lines):", 'INFO')
                for line in stdout_lines[-20:]:
                    print(f"  {line}")
            
            # Verify expected output exists
            if expected_output and not os.path.exists(expected_output):
                self.log(f"Expected output missing: {expected_output}", 'ERROR')
                self.results[step_id] = {
                    'status': 'FAILED',
                    'reason': f'Expected output not found: {expected_output}',
                    'command': cmd_str
                }
                return False
            
            self.results[step_id] = {
                'status': 'SUCCESS',
                'command': cmd_str,
                'output': expected_output
            }
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Step {step_id} FAILED (exit code {e.returncode})", 'ERROR')
            
            # Log error details
            self.log("STDOUT (last 50 lines):", 'ERROR')
            stdout_lines = e.stdout.strip().split('\n') if e.stdout else []
            for line in stdout_lines[-50:]:
                print(f"  {line}")
            
            self.log("STDERR (last 50 lines):", 'ERROR')
            stderr_lines = e.stderr.strip().split('\n') if e.stderr else []
            for line in stderr_lines[-50:]:
                print(f"  {line}")
            
            self.results[step_id] = {
                'status': 'FAILED',
                'exit_code': e.returncode,
                'command': cmd_str,
                'stdout_tail': stdout_lines[-50:] if stdout_lines else [],
                'stderr_tail': stderr_lines[-50:] if stderr_lines else []
            }
            return False
            
        except Exception as e:
            self.log(f"Step {step_id} FAILED with exception: {e}", 'ERROR')
            traceback.print_exc()
            
            self.results[step_id] = {
                'status': 'EXCEPTION',
                'exception': str(e),
                'command': cmd_str,
                'traceback': traceback.format_exc()
            }
            return False
    
    def step_a_train_city_teacher(self):
        """Step A: Train City Teacher"""
        if 'A' in self.skip_steps:
            self.log("Step A: SKIPPED (--skip-step A)", 'SKIP')
            self.results['A'] = {'status': 'SKIPPED'}
            return True
        
        cmd = [
            'python', 'train_teacher_city.py',
            '--json_path', self.json_path,
            '--data_root', self.data_root,
            '--target_name', self.target_name,
            '--num_scenes', str(self.num_scenes),
            '--num_cities', str(self.num_cities),
            '--batch_size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--output_dir', self.output_root
        ]
        
        return self.run_command(
            cmd,
            'A',
            self.STEP_NAMES['A'],
            expected_output=self.city_teacher_ckpt
        )
    
    def step_b_train_city2scene_teacher(self):
        """Step B: Train City2Scene Teacher with backbone freeze check"""
        if 'B' in self.skip_steps:
            self.log("Step B: SKIPPED (--skip-step B)", 'SKIP')
            self.results['B'] = {'status': 'SKIPPED'}
            return True
        
        # Verify prerequisite
        if not os.path.exists(self.city_teacher_ckpt):
            self.log(f"Prerequisite missing: {self.city_teacher_ckpt}", 'ERROR')
            self.results['B'] = {
                'status': 'FAILED',
                'reason': 'City teacher checkpoint not found (Step A failed or skipped)'
            }
            return False
        
        cmd = [
            'python', 'train_teacher_city2scene.py',
            '--json_path', self.json_path,
            '--data_root', self.data_root,
            '--target_name', self.target_name,
            '--city_teacher_ckpt', self.city_teacher_ckpt,
            '--num_scenes', str(self.num_scenes),
            '--num_cities', str(self.num_cities),
            '--batch_size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--output_dir', self.output_root
        ]
        
        return self.run_command(
            cmd,
            'B',
            self.STEP_NAMES['B'],
            expected_output=self.city2scene_ckpt
        )
    
    def step_c_train_scene_teacher(self):
        """Step C: Train Scene Teacher"""
        if 'C' in self.skip_steps:
            self.log("Step C: SKIPPED (--skip-step C)", 'SKIP')
            self.results['C'] = {'status': 'SKIPPED'}
            return True
        
        cmd = [
            'python', 'train_teacher_scene.py',
            '--json_path', self.json_path,
            '--data_root', self.data_root,
            '--target_name', 'global',
            '--num_scenes', str(self.num_scenes),
            '--batch_size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--output_dir', self.output_root
        ]
        
        return self.run_command(
            cmd,
            'C',
            self.STEP_NAMES['C'],
            expected_output=self.scene_ckpt
        )
    
    def step_d_dump_teacher_logits(self):
        """Step D: Dump teacher logits to cache"""
        if 'D' in self.skip_steps:
            self.log("Step D: SKIPPED (--skip-step D)", 'SKIP')
            self.results['D'] = {'status': 'SKIPPED'}
            return True
        
        # Verify prerequisites
        if not os.path.exists(self.city2scene_ckpt):
            self.log(f"Prerequisite missing: {self.city2scene_ckpt}", 'ERROR')
            self.results['D'] = {
                'status': 'FAILED',
                'reason': 'City2Scene teacher checkpoint not found (Step B failed or skipped)'
            }
            return False
        
        if not os.path.exists(self.scene_ckpt):
            self.log(f"Prerequisite missing: {self.scene_ckpt}", 'ERROR')
            self.results['D'] = {
                'status': 'FAILED',
                'reason': 'Scene teacher checkpoint not found (Step C failed or skipped)'
            }
            return False
        
        cmd = [
            'python', 'dump_teacher_logits.py',
            '--city2scene_teacher_ckpt', self.city2scene_ckpt,
            '--scene_teacher_ckpt', self.scene_ckpt,
            '--json_path', self.json_path,
            '--data_root', self.data_root,
            '--target_name', self.target_name,
            '--num_scenes', str(self.num_scenes),
            '--cache_root', self.cache_root,
            '--num_verify', '5'
        ]
        
        return self.run_command(
            cmd,
            'D',
            self.STEP_NAMES['D'],
            expected_output=self.cache_index
        )
    
    def step_e_verify_cache(self):
        """Step E: Verify cache exists with 100% hit rate"""
        if 'E' in self.skip_steps:
            self.log("Step E: SKIPPED (--skip-step E)", 'SKIP')
            self.results['E'] = {'status': 'SKIPPED'}
            return True
        
        # Verify prerequisite
        if not os.path.exists(self.cache_index):
            self.log(f"Prerequisite missing: {self.cache_index}", 'ERROR')
            self.results['E'] = {
                'status': 'FAILED',
                'reason': 'Cache index not found (Step D failed or skipped)'
            }
            return False
        
        self.print_separator()
        self.log(f"Step E: {self.STEP_NAMES['E']}", 'INFO')
        self.print_separator()
        
        try:
            # Load dataset using unified loader
            from utils.dataset_io import load_dataset
            from utils.path_key import make_cache_key
            
            self.log("Loading dataset with utils.dataset_io.load_dataset()", 'INFO')
            dataset = load_dataset(self.json_path)
            
            # Get train samples
            train_samples = dataset['train']
            self.log(f"Found {len(train_samples)} train samples", 'INFO')
            
            # Load cache index
            self.log(f"Loading cache index: {self.cache_index}", 'INFO')
            with open(self.cache_index, 'r') as f:
                cache_index = json.load(f)
            
            # Verify cache for both teachers
            missing_city2scene = []
            missing_scene = []
            
            for sample in train_samples:
                # Generate cache key using unified function
                cache_key = make_cache_key(sample, self.data_root)
                
                # Check city2scene cache
                if cache_key not in cache_index.get('city2scene', {}):
                    missing_city2scene.append(cache_key)
                
                # Check scene cache
                if cache_key not in cache_index.get('scene', {}):
                    missing_scene.append(cache_key)
            
            # Report results
            city2scene_hits = len(train_samples) - len(missing_city2scene)
            scene_hits = len(train_samples) - len(missing_scene)
            
            self.log(f"City2Scene cache: {city2scene_hits}/{len(train_samples)} hits "
                    f"({100.0 * city2scene_hits / len(train_samples):.1f}%)", 'INFO')
            self.log(f"Scene cache: {scene_hits}/{len(train_samples)} hits "
                    f"({100.0 * scene_hits / len(train_samples):.1f}%)", 'INFO')
            
            # Check for 100% hit rate
            if missing_city2scene or missing_scene:
                self.log("Cache verification FAILED: Missing cache entries", 'ERROR')
                if missing_city2scene:
                    self.log(f"Missing city2scene: {missing_city2scene[:5]}...", 'ERROR')
                if missing_scene:
                    self.log(f"Missing scene: {missing_scene[:5]}...", 'ERROR')
                
                self.results['E'] = {
                    'status': 'FAILED',
                    'reason': f'{len(missing_city2scene)} city2scene, {len(missing_scene)} scene missing',
                    'missing_city2scene_count': len(missing_city2scene),
                    'missing_scene_count': len(missing_scene)
                }
                return False
            
            self.log("Cache verification PASSED: 100% hit rate for both teachers", 'SUCCESS')
            self.results['E'] = {
                'status': 'SUCCESS',
                'city2scene_hits': city2scene_hits,
                'scene_hits': scene_hits,
                'total_samples': len(train_samples)
            }
            return True
            
        except Exception as e:
            self.log(f"Cache verification FAILED with exception: {e}", 'ERROR')
            traceback.print_exc()
            self.results['E'] = {
                'status': 'EXCEPTION',
                'exception': str(e),
                'traceback': traceback.format_exc()
            }
            return False
    
    def step_f_train_student(self):
        """Step F: Train student with city2scene KD (CE on source, KD on target)"""
        if 'F' in self.skip_steps:
            self.log("Step F: SKIPPED (--skip-step F)", 'SKIP')
            self.results['F'] = {'status': 'SKIPPED'}
            return True
        
        # Verify prerequisite
        if not os.path.exists(self.cache_index):
            self.log(f"Prerequisite missing: {self.cache_index}", 'ERROR')
            self.results['F'] = {
                'status': 'FAILED',
                'reason': 'Cache index not found (Step D failed or skipped)'
            }
            return False
        
        cmd = [
            'python', 'train_student_distill.py',
            '--json_path', self.json_path,
            '--data_root', self.data_root,
            '--target_name', self.target_name,
            '--num_scenes', str(self.num_scenes),
            '--teacher_mode', 'city2scene',
            '--cache_root', os.path.join(self.cache_root, f'A2{self.target_name}'),
            '--temperature', '4.0',
            '--kd_alpha', '0.7',
            '--batch_size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--output_dir', self.output_root
        ]
        
        return self.run_command(
            cmd,
            'F',
            self.STEP_NAMES['F'],
            expected_output=self.student_ckpt
        )
    
    def run_all(self):
        """Run all pipeline steps in sequence"""
        self.log("="*80, 'INFO')
        self.log("PIPELINE SANITY CHECK - START", 'INFO')
        self.log("="*80, 'INFO')
        self.log(f"JSON Path: {self.json_path}", 'INFO')
        self.log(f"Data Root: {self.data_root}", 'INFO')
        self.log(f"Output Root: {self.output_root}", 'INFO')
        self.log(f"Target: {self.target_name}, Scenes: {self.num_scenes}, Cities: {self.num_cities}", 'INFO')
        self.log(f"Batch Size: {self.batch_size}, Epochs: {self.epochs}", 'INFO')
        if self.skip_steps:
            self.log(f"Skipping steps: {', '.join(self.skip_steps)}", 'WARN')
        self.log("="*80, 'INFO')
        
        # Create output directory
        os.makedirs(self.output_root, exist_ok=True)
        
        # Run steps in sequence
        steps = [
            ('A', self.step_a_train_city_teacher),
            ('B', self.step_b_train_city2scene_teacher),
            ('C', self.step_c_train_scene_teacher),
            ('D', self.step_d_dump_teacher_logits),
            ('E', self.step_e_verify_cache),
            ('F', self.step_f_train_student)
        ]
        
        for step_id, step_func in steps:
            if not step_func():
                self.log(f"\n{'='*80}", 'ERROR')
                self.log(f"PIPELINE FAILED at Step {step_id}: {self.STEP_NAMES[step_id]}", 'ERROR')
                self.log(f"{'='*80}", 'ERROR')
                self.print_summary()
                return False
        
        # All steps passed
        self.print_separator('=', 100)
        self.log("ALL PIPELINE STEPS PASSED!", 'SUCCESS')
        self.print_separator('=', 100)
        self.print_summary()
        return True
    
    def print_summary(self):
        """Print summary of all steps"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{'='*100}")
        print("SUMMARY")
        print(f"{'='*100}")
        print(f"Total Time: {elapsed:.1f}s")
        print(f"Output Directory: {self.output_root}")
        print(f"\nStep Results:")
        
        for step_id in ['A', 'B', 'C', 'D', 'E', 'F']:
            result = self.results.get(step_id, {'status': 'NOT_RUN'})
            status = result['status']
            
            symbol = {
                'SUCCESS': '✓',
                'FAILED': '✗',
                'SKIPPED': '⊘',
                'EXCEPTION': '✗',
                'NOT_RUN': '-'
            }.get(status, '?')
            
            print(f"  [{symbol}] Step {step_id}: {self.STEP_NAMES[step_id]} - {status}")
            
            if status == 'FAILED' and 'reason' in result:
                print(f"      Reason: {result['reason']}")
            elif status == 'EXCEPTION' and 'exception' in result:
                print(f"      Exception: {result['exception']}")
        
        print(f"{'='*100}\n")
    
    def cleanup(self):
        """Clean up temporary outputs"""
        if os.path.exists(self.output_root):
            self.log(f"Cleaning up: {self.output_root}", 'INFO')
            shutil.rmtree(self.output_root)
            self.log("Cleanup complete", 'SUCCESS')


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Pipeline Sanity Check - CI/Local Gate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sanity check
  python check_pipeline_sanity.py --json_path data/sample.json --data_root data/
  
  # With cleanup after completion
  python check_pipeline_sanity.py --json_path data/sample.json --data_root data/ --cleanup
  
  # Skip certain steps (e.g., if already run)
  python check_pipeline_sanity.py --json_path data/sample.json --data_root data/ --skip-step A,B
        """
    )
    
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to dataset JSON file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory for data files')
    parser.add_argument('--output_root', type=str, default='outputs/_sanity',
                       help='Root directory for outputs (default: outputs/_sanity)')
    parser.add_argument('--num_scenes', type=int, default=10,
                       help='Number of scene classes (default: 10)')
    parser.add_argument('--num_cities', type=int, default=12,
                       help='Number of city classes (default: 12)')
    parser.add_argument('--target_name', type=str, default='b',
                       help='Target domain name (default: b)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs (default: 1)')
    parser.add_argument('--skip-step', type=str, default='',
                       help='Comma-separated list of steps to skip (e.g., A,B,C)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up output directory after completion')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.json_path):
        print(f"✗ Error: JSON file not found: {args.json_path}")
        sys.exit(1)
    
    if not os.path.exists(args.data_root):
        print(f"✗ Error: Data root not found: {args.data_root}")
        sys.exit(1)
    
    # Parse skip steps
    skip_steps = [s.strip().upper() for s in args.skip_step.split(',') if s.strip()]
    
    # Create checker
    checker = PipelineSanityChecker(
        json_path=args.json_path,
        data_root=args.data_root,
        output_root=args.output_root,
        num_scenes=args.num_scenes,
        num_cities=args.num_cities,
        target_name=args.target_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        skip_steps=skip_steps
    )
    
    # Run pipeline
    success = checker.run_all()
    
    # Cleanup if requested and successful
    if args.cleanup:
        if success:
            checker.cleanup()
        else:
            checker.log("Skipping cleanup due to failures (inspect outputs for debugging)", 'WARN')
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
