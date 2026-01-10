#!/usr/bin/env python3
"""
Unit tests for KD loss functions

Tests:
1. kd_loss with random logits
2. Gradient flow
3. Different loss types (kldiv vs soft_ce)
4. Confidence weighting strategies
5. Scheduling (warmup + ramp)
6. Combined student loss computation
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.kd import (
    kd_loss,
    compute_confidence_weights,
    compute_kd_weight,
    compute_student_loss,
    KDLossConfig
)


def test_kd_loss_basic():
    """Test basic KD loss computation"""
    print("\n" + "="*80)
    print("Test 1: Basic KD Loss")
    print("="*80)
    
    batch_size = 8
    num_classes = 10
    temperature = 4.0
    
    # Create random logits
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    
    # Compute KD loss
    loss = kd_loss(student_logits, teacher_logits, temperature=temperature)
    
    print(f"Student logits shape: {student_logits.shape}")
    print(f"Teacher logits shape: {teacher_logits.shape}")
    print(f"KD loss (KLDiv): {loss.item():.6f}")
    
    # Check properties
    assert loss.item() >= 0, "KD loss should be non-negative"
    assert not torch.isnan(loss), "KD loss contains NaN"
    assert not torch.isinf(loss), "KD loss contains Inf"
    
    print("✓ KD loss is valid (non-negative, no NaN/Inf)")
    
    return loss


def test_gradient_flow():
    """Test that gradients flow correctly"""
    print("\n" + "="*80)
    print("Test 2: Gradient Flow")
    print("="*80)
    
    batch_size = 8
    num_classes = 10
    
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    
    # Compute loss and backward
    loss = kd_loss(student_logits, teacher_logits)
    loss.backward()
    
    print(f"Loss value: {loss.item():.6f}")
    print(f"Student logits grad shape: {student_logits.grad.shape}")
    print(f"Student logits grad norm: {student_logits.grad.norm().item():.6f}")
    
    # Check gradients
    assert student_logits.grad is not None, "Student logits should have gradients"
    assert not torch.isnan(student_logits.grad).any(), "Gradients contain NaN"
    assert not torch.isinf(student_logits.grad).any(), "Gradients contain Inf"
    assert student_logits.grad.norm() > 0, "Gradients should be non-zero"
    
    print("✓ Gradients flow correctly (no NaN/Inf, non-zero)")
    
    return student_logits.grad


def test_loss_types():
    """Test different loss types (kldiv vs soft_ce)"""
    print("\n" + "="*80)
    print("Test 3: Different Loss Types")
    print("="*80)
    
    batch_size = 8
    num_classes = 10
    temperature = 4.0
    
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    
    # KLDiv loss
    loss_kldiv = kd_loss(student_logits, teacher_logits, 
                        temperature=temperature, loss_type='kldiv')
    
    # Soft CE loss
    loss_softce = kd_loss(student_logits, teacher_logits,
                         temperature=temperature, loss_type='soft_ce')
    
    print(f"KLDiv loss: {loss_kldiv.item():.6f}")
    print(f"Soft CE loss: {loss_softce.item():.6f}")
    
    # Both should be valid
    assert not torch.isnan(loss_kldiv), "KLDiv loss contains NaN"
    assert not torch.isnan(loss_softce), "Soft CE loss contains NaN"
    assert loss_kldiv.item() >= 0, "KLDiv loss should be non-negative"
    assert loss_softce.item() >= 0, "Soft CE loss should be non-negative"
    
    # Should be similar but not identical
    diff = abs(loss_kldiv.item() - loss_softce.item())
    print(f"Difference: {diff:.6f}")
    
    print("✓ Both loss types work correctly")
    
    return loss_kldiv, loss_softce


def test_confidence_weighting():
    """Test confidence weighting strategies"""
    print("\n" + "="*80)
    print("Test 4: Confidence Weighting")
    print("="*80)
    
    batch_size = 16
    num_classes = 10
    
    # Create teacher logits with varying confidence
    torch.manual_seed(42)
    teacher_logits = torch.randn(batch_size, num_classes)
    
    # Test 'none' weighting
    weights_none, stats_none = compute_confidence_weights(
        teacher_logits, weighting='none'
    )
    print(f"\nWeighting: none")
    print(f"  Weights: {weights_none[:4].tolist()}")
    print(f"  Coverage: {stats_none['coverage']:.2%}")
    print(f"  Avg confidence: {stats_none['avg_confidence']:.4f}")
    assert torch.all(weights_none == 1.0), "All weights should be 1.0"
    
    # Test 'pmax' weighting
    weights_pmax, stats_pmax = compute_confidence_weights(
        teacher_logits, weighting='pmax'
    )
    print(f"\nWeighting: pmax")
    print(f"  Weights: {weights_pmax[:4].tolist()}")
    print(f"  Coverage: {stats_pmax['coverage']:.2%}")
    print(f"  Avg confidence: {stats_pmax['avg_confidence']:.4f}")
    assert torch.all(weights_pmax >= 0) and torch.all(weights_pmax <= 1), \
        "Pmax weights should be in [0, 1]"
    
    # Test 'threshold' weighting
    threshold = 0.5
    weights_thresh, stats_thresh = compute_confidence_weights(
        teacher_logits, weighting='threshold', threshold=threshold
    )
    print(f"\nWeighting: threshold (>= {threshold})")
    print(f"  Weights: {weights_thresh[:4].tolist()}")
    print(f"  Coverage: {stats_thresh['coverage']:.2%}")
    print(f"  Avg confidence: {stats_thresh['avg_confidence']:.4f}")
    assert torch.all((weights_thresh == 0) | (weights_thresh == 1)), \
        "Threshold weights should be binary"
    
    print("\n✓ All weighting strategies work correctly")
    
    return weights_none, weights_pmax, weights_thresh


def test_scheduling():
    """Test warmup and ramp scheduling"""
    print("\n" + "="*80)
    print("Test 5: Warmup + Ramp Scheduling")
    print("="*80)
    
    warmup_epochs = 5
    kd_ramp_epochs = 10
    target_alpha = 0.7
    
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Ramp epochs: {kd_ramp_epochs}")
    print(f"Target alpha: {target_alpha}")
    print()
    
    epochs_to_test = [0, 2, 4, 5, 7, 10, 14, 15, 20]
    
    for epoch in epochs_to_test:
        kd_alpha = compute_kd_weight(
            epoch, warmup_epochs, kd_ramp_epochs, target_alpha
        )
        
        # Check expected behavior
        if epoch < warmup_epochs:
            expected = 0.0
            phase = "Warmup"
        elif epoch < warmup_epochs + kd_ramp_epochs:
            ramp_progress = (epoch - warmup_epochs) / kd_ramp_epochs
            expected = target_alpha * ramp_progress
            phase = "Ramp"
        else:
            expected = target_alpha
            phase = "Full"
        
        print(f"Epoch {epoch:2d} ({phase:6s}): kd_alpha = {kd_alpha:.4f} "
              f"(expected: {expected:.4f})")
        
        assert abs(kd_alpha - expected) < 1e-6, \
            f"KD weight mismatch at epoch {epoch}"
    
    print("\n✓ Scheduling works correctly")


def test_combined_student_loss():
    """Test combined student loss computation"""
    print("\n" + "="*80)
    print("Test 6: Combined Student Loss")
    print("="*80)
    
    batch_size = 16
    num_classes = 10
    
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test with default config
    config = KDLossConfig(
        temperature=4.0,
        kd_alpha=0.7,
        loss_type='kldiv',
        kd_weighting='pmax',
        confidence_threshold=0.0,
        warmup_epochs=2,
        kd_ramp_epochs=3
    )
    
    print("Testing different epochs:")
    print()
    
    for epoch in [0, 1, 2, 3, 5, 10]:
        loss, stats = compute_student_loss(
            student_logits,
            teacher_logits,
            targets,
            config,
            current_epoch=epoch
        )
        
        print(f"Epoch {epoch}:")
        print(f"  Total loss: {stats['total_loss']:.6f}")
        print(f"  CE loss: {stats['ce_loss']:.6f}")
        print(f"  KD loss: {stats['kd_loss']:.6f}")
        print(f"  KD weight: {stats['kd_weight']:.4f}")
        print(f"  CE weight: {stats['ce_weight']:.4f}")
        print(f"  Coverage: {stats['coverage']:.2%}")
        print(f"  Avg confidence: {stats['avg_confidence']:.4f}")
        
        # Validate
        assert not torch.isnan(loss), f"Loss is NaN at epoch {epoch}"
        assert not torch.isinf(loss), f"Loss is Inf at epoch {epoch}"
        assert loss.item() >= 0, f"Loss is negative at epoch {epoch}"
        
        # Test backward
        if student_logits.grad is not None:
            student_logits.grad.zero_()
        loss.backward(retain_graph=True)
        
        assert student_logits.grad is not None, "Gradients should exist"
        assert not torch.isnan(student_logits.grad).any(), "Gradients contain NaN"
        
        print()
    
    print("✓ Combined student loss works correctly")


def test_domain_masking():
    """Test domain masking in combined loss"""
    print("\n" + "="*80)
    print("Test 7: Domain Masking")
    print("="*80)
    
    batch_size = 16
    num_classes = 10
    
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Create domain mask (e.g., first half source, second half target)
    domain_mask = torch.zeros(batch_size, dtype=torch.bool)
    domain_mask[batch_size//2:] = True  # Only apply KD to second half
    
    config = KDLossConfig(
        temperature=4.0,
        kd_alpha=0.7,
        kd_weighting='none'
    )
    
    # Without mask
    loss_no_mask, stats_no_mask = compute_student_loss(
        student_logits, teacher_logits, targets, config, current_epoch=10
    )
    
    # With mask
    loss_with_mask, stats_with_mask = compute_student_loss(
        student_logits, teacher_logits, targets, config, current_epoch=10,
        domain_mask=domain_mask
    )
    
    print("Without domain mask:")
    print(f"  Coverage: {stats_no_mask['coverage']:.2%}")
    print(f"  KD loss: {stats_no_mask['kd_loss']:.6f}")
    
    print("\nWith domain mask (50% samples):")
    print(f"  Coverage: {stats_with_mask['coverage']:.2%}")
    print(f"  KD loss: {stats_with_mask['kd_loss']:.6f}")
    
    # Coverage should be reduced with mask
    assert stats_with_mask['coverage'] < stats_no_mask['coverage'], \
        "Coverage should be lower with domain mask"
    
    print("\n✓ Domain masking works correctly")


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("\n" + "="*80)
    print("Test 8: Numerical Stability")
    print("="*80)
    
    batch_size = 8
    num_classes = 10
    
    # Test with very confident teacher (large logits)
    torch.manual_seed(42)
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes) * 10  # Large logits
    
    loss = kd_loss(student_logits, teacher_logits, temperature=4.0)
    
    print(f"Large teacher logits:")
    print(f"  Teacher logits range: [{teacher_logits.min():.2f}, {teacher_logits.max():.2f}]")
    print(f"  KD loss: {loss.item():.6f}")
    print(f"  Valid: {not torch.isnan(loss) and not torch.isinf(loss)}")
    
    assert not torch.isnan(loss), "Loss is NaN with large logits"
    assert not torch.isinf(loss), "Loss is Inf with large logits"
    
    print("\n✓ Numerically stable")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*80)
    print("KD LOSS UNIT TESTS")
    print("="*80)
    
    try:
        test_kd_loss_basic()
        test_gradient_flow()
        test_loss_types()
        test_confidence_weighting()
        test_scheduling()
        test_combined_student_loss()
        test_domain_masking()
        test_numerical_stability()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED ✗: {str(e)}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
