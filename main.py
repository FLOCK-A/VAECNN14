"""
训练主流程脚本
使用P参数控制训练流程，损失函数使用objectives模块，超参数来自config模块
"""
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from config.config import config
from models.classifier import Cnn14Classifier
from data.dataloader import get_dataloader
from objectives import compute_batch_loss
from augmentations import ASCStrongAugment
import json
import math


def create_P_params():
    """
    创建P参数字典，用于控制训练流程
    """
    P = {
        # 实验模式：'classification' 表示仅分类，'adaptation' 表示域适应
        'exp_mode': config.EXP_MODE,  # 具体作用：决定是否使用域适应损失
        
        # 域适应方法：指定使用的域适应算法
        'adapt_method': config.ADAPT_METHOD,  # 具体作用：指定域适应损失函数类型
        
        # 用户定义的域适应损失权重
        'user_lamda': config.USER_LAMDA,  # 具体作用：控制域适应损失在总损失中的权重
        
        # 设备分类损失权重
        'lambda_dev': getattr(config, "LAMBDA_DEV", 0.1),  # 具体作用：控制设备预测损失在总损失中的权重
        
        # 设备数量
        'num_devices': getattr(config, "NUM_DEVICES", 6),  # 具体作用：设备类别数量
        
        # 正交/去相关损失权重
        'lambda_orth': getattr(config, "LAMBDA_ORTH", 0.0),  # 具体作用：控制正交损失在总损失中的权重
        
        # 语义-设备独立性损失权重
        'lambda_indep': getattr(config, "LAMBDA_INDEP", 0.0),  # 具体作用：控制独立性损失在总损失中的权重
        
        # 中心矩匹配的最高阶数（用于CMD和HoMM方法）
        'highest_moment': config.HIGHEST_MOMENT,  # 具体作用：指定计算中心矩匹配的最高阶数
        
        # 几何适应的距离度量类型（用于geo_adapt方法）
        'dist_metric_type': config.DIST_METRIC_TYPE,  # 具体作用：指定geo_adapt方法的距离度量类型
        
        # 行列式阈值（用于geo_adapt方法）
        'det_thr': config.DET_THR,  # 具体作用：决定是否应用域适应损失的阈值
        
        # 训练相关参数
        'num_epochs': config.NUM_EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'batch_size': config.BATCH_SIZE,
        'warmup_epochs': config.WARMUP_EPOCHS,
        'min_lr': config.MIN_LR,
    }
    return P


def load_dataset():
    """
    加载数据集，从JSON文件中读取源域和目标域数据
    """
    # 从JSON文件加载数据集信息
    with open(config.DATASET_JSON, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # 创建源域数据加载器（train作为源域）
    source_loader = get_dataloader(
        samples=dataset_info['train'],
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE//2,
        shuffle=True
    )
    
    # 创建目标域数据加载器（val作为目标域）
    target_loader = get_dataloader(
        samples=dataset_info['val'],
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE//2,
        shuffle=True
    )
    
    # 创建验证数据加载器（使用val数据）
    val_loader = get_dataloader(
        samples=dataset_info['val'],
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # 创建测试数据加载器（使用test数据）
    test_loader = get_dataloader(
        samples=dataset_info['test'],
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return source_loader, target_loader, val_loader, test_loader

'''
def create_mixed_batch(source_batch, target_batch):
    """
    将源域和目标域的批次合并成一个批次，确保每个批次中一半是源域一半是目标域
    """
    # 获取源域和目标域数据
    source_features = source_batch['features']
    source_labels = source_batch['label']
    source_domains = source_batch['domain']
    
    target_features = target_batch['features']
    target_labels = target_batch['label']
    target_domains = target_batch['domain']
    
    # 计算每种域的数据量
    source_size = source_features.size(0)
    target_size = target_features.size(0)
    
    # 计算每种域需要取的数量（确保总数不超过配置的批次大小）
    total_samples = min(source_size + target_size, config.BATCH_SIZE)
    half_batch = total_samples // 2
    
    # 根据实际可用数据调整每个域的样本数
    src_samples = min(source_size, half_batch)
    trg_samples = min(target_size, half_batch)
    
    # 选择样本
    mixed_features = torch.cat([source_features[:src_samples], target_features[:trg_samples]], dim=0)
    mixed_labels = torch.cat([source_labels[:src_samples], target_labels[:trg_samples]], dim=0)
    mixed_domains = torch.cat([source_domains[:src_samples], target_domains[:trg_samples]], dim=0)
    
    return {
        'features': mixed_features,
        'labels': mixed_labels,
        'domains': mixed_domains,
        'num_src_samples': src_samples  # 源域样本数量
    }
'''
def create_mixed_batch(source_batch, target_batch):
    """
    直接将源域批次和目标域批次串联（源在前，目标在后）。
    假设 source_batch 和 target_batch 的键与项目中一致：
    'features', 'label', 'domain'。返回键与训练流程一致：
    'features','labels','domains','num_src_samples'。
    """
    source_features = source_batch['features']
    source_labels = source_batch['label']
    source_domains = source_batch['domain']

    target_features = target_batch['features']
    target_labels = target_batch['label']
    target_domains = target_batch['domain']

    src_n = source_features.size(0)
    trg_n = target_features.size(0)

    mixed_features = torch.cat([source_features, target_features], dim=0)
    mixed_labels = torch.cat([source_labels, target_labels], dim=0)
    mixed_domains = torch.cat([source_domains, target_domains], dim=0)

    return {
        'features': mixed_features,
        'labels': mixed_labels,
        'domains': mixed_domains,
        'num_src_samples': src_n
    }


def train_epoch(model, source_loader, target_loader, optimizer, P, current_epoch):
    """
    单个训练周期
    """
    model.train()
    total_loss = 0.0
    total_class_loss = 0.0
    total_da_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_steps = 0  # 真实步数计数器
    
    # 实例化增强器
    augmenter = ASCStrongAugment(
        num_classes=getattr(config, "NUM_CLASSES", 10),
        enable=getattr(config, "AUG_ENABLE", True),
    
        p_time_roll=getattr(config, "AUG_P_TIME_ROLL", 0.5),
    
        p_specaug=getattr(config, "AUG_P_SPECAUG", 0.8),
        freq_mask_ratio=getattr(config, "AUG_FREQ_MASK_RATIO", 0.15),
        time_mask_ratio=getattr(config, "AUG_TIME_MASK_RATIO", 0.15),
        num_freq_masks=getattr(config, "AUG_NUM_FREQ_MASKS", 2),
        num_time_masks=getattr(config, "AUG_NUM_TIME_MASKS", 2),
    
        p_filter=getattr(config, "AUG_P_FILTER", 0.5),
        filter_n_band=getattr(config, "AUG_FILTER_N_BAND", (4, 8)),
        filter_db=getattr(config, "AUG_FILTER_DB", (-6.0, 6.0)),
        filter_mode=getattr(config, "AUG_FILTER_MODE", "add"),
    
        p_fms=getattr(config, "AUG_P_FMS", 0.5),
        fms_alpha=getattr(config, "AUG_FMS_ALPHA", 0.6),
        fms_mix=getattr(config, "AUG_FMS_MIX", "crossdomain"),
    
        p_mixup=getattr(config, "AUG_P_MIXUP", 1.0),
        mixup_alpha=getattr(config, "AUG_MIXUP_ALPHA", 0.4),
    )
    
    # 创建迭代器
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    for batch_idx in range(min(len(source_loader), len(target_loader))):
        try:
            # 获取源域和目标域批次
            source_batch = next(source_iter)
            target_batch = next(target_iter)
        except StopIteration:
            break
        
        # 创建混合批次（一半源域，一半目标域）
        mixed_batch = create_mixed_batch(source_batch, target_batch)
        
        features = mixed_batch['features']
        labels = mixed_batch['labels']
        domains = mixed_batch['domains']          # 新增
        num_src_samples = mixed_batch['num_src_samples']
        
        # 将数据移到GPU（如果可用）
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            domains = domains.cuda()
        
        # ===== 增强：只在训练阶段做 =====
        labels_onehot = None
        if augmenter is not None and getattr(config, "AUG_ENABLE", True):
            if P['exp_mode'] == 'adaptation':
                features, labels_onehot = augmenter(features, labels, num_src_samples)
            else:
                # 分类模式：全 batch 都是"有标签"的
                features, labels_onehot = augmenter(features, labels, features.size(0))
        
        # 准备batch字典，用于损失计算
        batch_data = {
            'features': features,
            'labels': labels
        }
        
        # 如果是域适应模式，需要额外的数据
        if P['exp_mode'] == 'adaptation':
            # 在warmup期间不使用域适应方法
            if current_epoch < config.WARMUP_EPOCHS:
                # 只进行分类训练，不使用域适应
                logits, dev_logits, z_s, z_d = model(features, return_device_logits=True, return_latents=True)
                
                # 准备batch字典（不包含域适应相关数据）
                batch_for_loss = {
                    'logits': logits,
                    'labels': labels,
                    'num_src_samples': num_src_samples,
                    'domains': domains,
                    'dev_logits': dev_logits,
                    'z_s': z_s,
                    'z_d': z_d,
                }
                if labels_onehot is not None:
                    batch_for_loss['labels_onehot'] = labels_onehot
            else:
                # 获取模型的输出和用于域适应的特征
                logits, extracted_features, dev_logits, z_s, z_d = model(
                    features, return_adapt_features=True, return_device_logits=True, return_latents=True
                )
                
                # 准备batch字典
                batch_for_loss = {
                    'logits': logits,
                    'labels': labels,
                    'num_src_samples': num_src_samples,
                    'latent_feat': extracted_features,
                    'adapt_method': P['adapt_method'],
                    'domains': domains,
                    'dev_logits': dev_logits,
                    'z_s': z_s,
                    'z_d': z_d,
                }
                if labels_onehot is not None:
                    batch_for_loss['labels_onehot'] = labels_onehot
        else:
            # 分类模式
            logits = model(features)
            batch_for_loss = {
                'logits': logits,
                'labels': labels
            }
        
        # 计算损失
        batch_for_loss = compute_batch_loss(batch_for_loss, P)
        loss = batch_for_loss['loss_tensor']
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 记录分类损失和域适应损失
        if 'cs_loss_np' in batch_for_loss:
            total_class_loss += batch_for_loss['cs_loss_np']
        if 'da_loss_np' in batch_for_loss and batch_for_loss['da_loss_np'] is not None:
            total_da_loss += batch_for_loss['da_loss_np']
        
        # 计算准确率（仅对源域数据计算，因为目标域标签可能未知）
        _, predicted = torch.max(logits.data, 1)
        total_correct += (predicted[:num_src_samples] == labels[:num_src_samples]).sum().item()
        total_samples += num_src_samples
        
        n_steps += 1  # 增加真实步数计数
        
        if batch_idx % 10 == 0:
            # 打印当前批次的损失信息，包括设备损失
            print_str = f'Batch {batch_idx}, Loss: {loss.item():.4f}'
            if 'cs_loss_np' in batch_for_loss:
                print_str += f', Cs Loss: {batch_for_loss["cs_loss_np"]:.4f}'
            if 'da_loss_np' in batch_for_loss and batch_for_loss['da_loss_np'] is not None:
                print_str += f', DA Loss: {batch_for_loss["da_loss_np"]:.4f}'
            if 'dev_loss_np' in batch_for_loss:
                print_str += f', Dev Loss: {batch_for_loss["dev_loss_np"]:.4f}'
            if 'orth_loss_np' in batch_for_loss:
                print_str += f', Orth Loss: {batch_for_loss["orth_loss_np"]:.4f}'
            if 'indep_loss_np' in batch_for_loss:
                print_str += f', Indep Loss: {batch_for_loss["indep_loss_np"]:.4f}'
            print(print_str)
    
    avg_loss = total_loss / n_steps if n_steps > 0 else 0.0
    avg_class_loss = total_class_loss / n_steps if n_steps > 0 else 0.0
    avg_da_loss = total_da_loss / n_steps if n_steps > 0 else 0.0
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_class_loss, avg_da_loss, accuracy


def validate(model, val_loader, P):
    """
    验证模型
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features']
            labels = batch['label']
            
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            # 前向传播
            logits = model(features)
            
            # 准备batch字典
            batch_for_loss = {
                'logits': logits,
                'labels': labels
            }
            
            # 计算损失
            batch_for_loss = compute_batch_loss(batch_for_loss, P)
            loss = batch_for_loss['loss_tensor']
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def test(model, test_loader, P):
    """
    测试模型性能
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            labels = batch['label']
            
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            # 前向传播
            logits = model(features)
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = 100.0 * total_correct / total_samples
    
    return accuracy


def main():
    """
    主训练流程
    """
    print("开始训练流程...")
    
    # 创建P参数
    P = create_P_params()
    print(f"实验模式: {P['exp_mode']}")
    print(f"域适应方法: {P['adapt_method']}")
    print(f"域适应损失权重: {P['user_lamda']}")
    print(f"Warmup轮数: {P['warmup_epochs']}")
    
    # 加载数据
    print("加载数据集...")
    source_loader, target_loader, val_loader, test_loader = load_dataset()
    
    # 创建模型
    print("创建模型...")
    # 假设有10个类别（根据数据集sample_dataset.json推断）
    num_classes = 10
    model = Cnn14Classifier(classes_num=num_classes, checkpoint_path=config.PRETRAINED_CHECKPOINT_PATH, use_projection_head=config.USE_PROJECTION_HEAD, adapt_dim=128)
    
    # 如果GPU可用，将模型移到GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 创建优化器
    optimizer = Adam(
        model.parameters(),
        lr=P['learning_rate'],
        weight_decay=P['weight_decay']
    )
    
    # 学习率调度器 - Warmup + 余弦退火
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr=1e-6):
        def lr_lambda(current_epoch):
            if current_epoch < num_warmup_epochs:
                # 线性warmup，与train.py保持一致
                return float(current_epoch + 1) / float(num_warmup_epochs + 1)
            else:
                # 余弦退火阶段，与train.py保持一致
                import math
                return 0.5 * (1 + math.cos(math.pi * (current_epoch - num_warmup_epochs) / (num_training_epochs - num_warmup_epochs)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs=config.WARMUP_EPOCHS, num_training_epochs=P['num_epochs'], min_lr=config.MIN_LR)
    
    # 训练统计
    train_losses = []
    train_class_losses = []
    train_da_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 用于记录最佳模型的指标
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    
    print("开始训练...")
    for epoch in range(P['num_epochs']):
        print(f"Epoch {epoch+1}/{P['num_epochs']}")
        
        # 训练
        train_loss, train_class_loss, train_da_loss, train_acc = train_epoch(model, source_loader, target_loader, optimizer, P, epoch)
        train_losses.append(train_loss)
        train_class_losses.append(train_class_loss)
        train_da_losses.append(train_da_loss)
        train_accuracies.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, P)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 打印每轮的详细信息
        print(f"Train Loss: {train_loss:.4f}, Train Class Loss: {train_class_loss:.4f}, Train DA Loss: {train_da_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 检查是否为最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            print("保存最佳模型")
    
    # 保存最终模型
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, 'final_model.pth'))
    print("保存最终模型")
    
    # 打印最佳模型的指标
    print(f"最佳模型在第 {best_epoch+1} 轮")
    print(f"最佳模型 Train Acc: {best_train_acc:.2f}%")
    print(f"最佳模型 Val Acc: {best_val_acc:.2f}%")
    
    # 使用测试集评估最佳模型
    print("加载最佳模型进行测试...")
    best_model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        if torch.cuda.is_available():
            model = model.cuda()
        test_accuracy = test(model, test_loader, P)
        print(f"测试集准确率: {test_accuracy:.2f}%")
    else:
        print("未找到最佳模型文件")
    
    print("训练完成!")


if __name__ == '__main__':
    main()