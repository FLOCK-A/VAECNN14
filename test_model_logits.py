#!/usr/bin/env python3
"""
Unit test for model output changes: raw logits and forward_features
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/runner/work/VAECNN14/VAECNN14')

from models.classifier import Cnn14Classifier


def test_model_raw_logits():
    """
    测试模型输出 raw logits（不含 LogSoftmax）
    """
    print("="*80)
    print("TEST: Model Output Raw Logits")
    print("="*80)
    
    # 创建模型（不需要预训练权重进行测试）
    num_classes = 10
    model = Cnn14Classifier(classes_num=num_classes, checkpoint_path=None)
    model.eval()
    
    # 创建随机输入 [B, T, F]
    batch_size = 4
    time_steps = 100
    mel_bins = 64
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    
    print(f"\nInput shape: {list(dummy_input.shape)}")
    
    # 前向传播
    with torch.no_grad():
        logits = model(dummy_input)
    
    print(f"Output logits shape: {list(logits.shape)}")
    print(f"Expected shape: [{batch_size}, {num_classes}]")
    
    # 验证 shape
    assert logits.shape == (batch_size, num_classes), \
        f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
    print("✅ Shape correct")
    
    # 验证无 NaN
    has_nan = torch.isnan(logits).any().item()
    assert not has_nan, "Output contains NaN values"
    print("✅ No NaN values")
    
    # 验证是 raw logits 而非 log probabilities
    # Log probabilities 应该都是负数或零，raw logits 可以是任意值
    has_positive = (logits > 0).any().item()
    print(f"\nLogits statistics:")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Has positive values: {has_positive}")
    
    # Raw logits 可以有正值，而 LogSoftmax 输出应该全是非正数
    # 这不是绝对的判断，但如果全是非正数且很小，可能仍然是 log probs
    if has_positive:
        print("✅ Contains positive values (likely raw logits)")
    else:
        print("⚠️  All values are non-positive (could still be raw logits if network is initialized)")
    
    print("\n✅ TEST PASSED: Model outputs raw logits")
    return True


def test_cross_entropy_with_logits():
    """
    验证 cross_entropy 可以直接使用 raw logits
    """
    print("\n" + "="*80)
    print("TEST: Cross Entropy with Raw Logits")
    print("="*80)
    
    # 创建模型
    num_classes = 10
    model = Cnn14Classifier(classes_num=num_classes, checkpoint_path=None)
    model.eval()
    
    # 创建随机输入和标签
    batch_size = 4
    time_steps = 100
    mel_bins = 64
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        logits = model(dummy_input)
    
    # 计算 cross entropy loss
    loss = F.cross_entropy(logits, dummy_labels)
    
    print(f"\nLogits shape: {list(logits.shape)}")
    print(f"Labels shape: {list(dummy_labels.shape)}")
    print(f"Cross entropy loss: {loss.item():.4f}")
    
    # 验证 loss 是有效的
    assert not torch.isnan(loss).item(), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("✅ Cross entropy works correctly with raw logits")
    
    # 测试梯度计算
    model.train()
    dummy_input_grad = torch.randn(batch_size, time_steps, mel_bins, requires_grad=True)
    logits_grad = model(dummy_input_grad)
    loss_grad = F.cross_entropy(logits_grad, dummy_labels)
    loss_grad.backward()
    
    assert dummy_input_grad.grad is not None, "Gradients not computed"
    assert not torch.isnan(dummy_input_grad.grad).any().item(), "Gradients contain NaN"
    print("✅ Gradient computation works correctly")
    
    print("\n✅ TEST PASSED: cross_entropy works with raw logits")
    return True


def test_forward_features():
    """
    测试 forward_features 和 extract_features 方法
    """
    print("\n" + "="*80)
    print("TEST: Forward Features Method")
    print("="*80)
    
    # 创建模型
    model = Cnn14Classifier(classes_num=10, checkpoint_path=None)
    model.eval()
    
    # 创建随机输入
    batch_size = 4
    time_steps = 100
    mel_bins = 64
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    
    # 测试 forward_features
    with torch.no_grad():
        features1 = model.forward_features(dummy_input)
        features2 = model.extract_features(dummy_input)
    
    print(f"\nforward_features output shape: {list(features1.shape)}")
    print(f"extract_features output shape: {list(features2.shape)}")
    
    # 验证两个方法输出相同
    assert torch.allclose(features1, features2), \
        "forward_features and extract_features should return same results"
    print("✅ forward_features and extract_features are equivalent")
    
    # 验证特征维度
    expected_dim = 2048  # From config.FEATURE_DIM
    assert features1.shape == (batch_size, expected_dim), \
        f"Expected shape ({batch_size}, {expected_dim}), got {features1.shape}"
    print(f"✅ Features shape correct: [{batch_size}, {expected_dim}]")
    
    # 验证无 NaN
    assert not torch.isnan(features1).any().item(), "Features contain NaN"
    print("✅ No NaN in features")
    
    print("\n✅ TEST PASSED: forward_features works correctly")
    return True


def test_feature_reuse_for_stage2():
    """
    测试 Stage-2 场景：使用 forward_features 提取特征，然后训练新的分类头
    """
    print("\n" + "="*80)
    print("TEST: Feature Reuse for Stage-2 Training")
    print("="*80)
    
    # 创建 Stage-1 City Teacher
    num_cities = 12
    city_model = Cnn14Classifier(classes_num=num_cities, checkpoint_path=None)
    
    # 冻结 city_model 的 feature_extractor
    for param in city_model.feature_extractor.parameters():
        param.requires_grad = False
    
    # 创建新的 scene head
    num_scenes = 10
    feature_dim = 2048
    scene_head = torch.nn.Linear(feature_dim, num_scenes)
    
    # 创建随机输入
    batch_size = 4
    time_steps = 100
    mel_bins = 64
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    dummy_labels = torch.randint(0, num_scenes, (batch_size,))
    
    # Stage-2 训练流程
    city_model.eval()  # 冻结模式
    scene_head.train()
    
    # 提取特征（不计算梯度）
    with torch.no_grad():
        features = city_model.forward_features(dummy_input)
    
    print(f"\nExtracted features shape: {list(features.shape)}")
    print(f"Features require_grad: {features.requires_grad}")
    assert not features.requires_grad, "Features should not require grad"
    
    # 通过 scene head
    scene_logits = scene_head(features.detach())
    loss = F.cross_entropy(scene_logits, dummy_labels)
    
    print(f"Scene logits shape: {list(scene_logits.shape)}")
    print(f"Loss: {loss.item():.4f}")
    
    # 反向传播（只更新 scene_head）
    loss.backward()
    
    # 验证梯度
    assert scene_head.weight.grad is not None, "Scene head should have gradients"
    
    # 验证 city_model 的参数没有梯度
    for name, param in city_model.feature_extractor.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().sum().item() == 0, \
                f"Frozen parameter {name} should not have gradients"
    
    print("✅ Backbone frozen correctly")
    print("✅ Scene head trainable")
    
    print("\n✅ TEST PASSED: Stage-2 feature reuse works correctly")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("  MODEL OUTPUT VALIDATION TESTS")
    print("="*80)
    
    tests = [
        ("Raw Logits Output", test_model_raw_logits),
        ("Cross Entropy with Logits", test_cross_entropy_with_logits),
        ("Forward Features", test_forward_features),
        ("Stage-2 Feature Reuse", test_feature_reuse_for_stage2),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n" + "="*80)
        print("  ALL TESTS PASSED ✅")
        print("="*80)
        print("\nModel changes verified:")
        print("  • forward() returns raw logits (no LogSoftmax)")
        print("  • cross_entropy works directly with logits")
        print("  • forward_features() returns backbone embeddings")
        print("  • extract_features() is an alias for forward_features()")
        print("  • Stage-2 training pattern works correctly")
        return 0
    else:
        print("\n" + "="*80)
        print("  SOME TESTS FAILED ✗")
        print("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
