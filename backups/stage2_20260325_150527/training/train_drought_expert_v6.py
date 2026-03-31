"""
专家级五分类训练脚本 v6
- 专门针对五分类任务优化
- 更强的类别不平衡处理
- 自适应模型复杂度
- 多阶段训练策略
"""

from datasets.dataset_drought import build_datasets_balanced
from models.net_drought import DroughtClassifier
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========== 专家级配置 ==========
CSV_PATH = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'

# 自适应训练参数
BATCH_SIZE = 4
EPOCHS = 80  # 增加训练轮数
LR_STAGE1 = 1e-3  # 第一阶段高学习率
LR_STAGE2 = 5e-5  # 第二阶段低学习率
NUM_CLASSES = 5
NUM_WORKERS = 4
RANDOM_SEED = 123
TEST_SIZE = 0.2

# 专家级正则化
WEIGHT_DECAY = 5e-4  # 中等权重衰减
LABEL_SMOOTHING = 0.2  # 更强的标签平滑
GRAD_CLIP = 0.5  # 更严格的梯度裁剪

# 多阶段训练
STAGE1_EPOCHS = 20  # 第一阶段：快速收敛
STAGE2_EPOCHS = 60  # 第二阶段：精细调优

SAVE_DIR = './models_expert'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_best_expert.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_enhanced_sampler(dataset, labels):
    """增强版均衡采样器（针对五分类优化）"""
    # 计算每个类别的样本权重（更强的少数类别权重）
    class_counts = np.bincount(labels)

    # 使用平方根权重，更温和地处理不平衡
    class_weights = 1. / np.sqrt(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]

    # 创建加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


def adaptive_model_complexity(num_classes):
    """根据类别数量自适应模型复杂度"""
    if num_classes == 2:
        return 32, [2, 2], [4, 4, 4]  # 二分类：简单模型
    elif num_classes == 5:
        return 48, [3, 3], [4, 4, 4]  # 五分类：中等复杂度（修正维度）
    else:
        return 64, [4, 4], [8, 8, 8]  # 多分类：复杂模型


def train_one_epoch(model, loader, criterion, optimizer, device, stage):
    """训练一个 epoch（分阶段策略）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Training Stage {stage}", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)

        # 确保标签是 Long 类型
        if labels.dtype == torch.float32 or labels.dtype == torch.float64:
            labels = labels.long()

        optimizer.zero_grad()
        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪（分阶段策略）
        clip_value = GRAD_CLIP if stage == 1 else GRAD_CLIP * 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, detailed=False):
    """验证模型（支持详细分析）"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 每个类别的准确率统计
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # 混淆矩阵统计
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    pbar = tqdm(loader, desc="Validation", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 统计每个类别的准确率
        for i in range(NUM_CLASSES):
            class_mask = (labels == i)
            class_total[i] += class_mask.sum().item()
            class_correct[i] += (predicted[class_mask] ==
                                 labels[class_mask]).sum().item()

        # 构建混淆矩阵
        if detailed:
            for true_label, pred_label in zip(labels.cpu(), predicted.cpu()):
                confusion_matrix[true_label, pred_label] += 1

    epoch_loss = total_loss / len(loader)
    epoch_acc = 100. * correct / total

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0.0)

    if detailed:
        return epoch_loss, epoch_acc, class_accuracies, confusion_matrix
    else:
        return epoch_loss, epoch_acc, class_accuracies


def main():
    print('=' * 60)
    print('专家级五分类训练 v6')
    print('=' * 60)

    print(f'Device: {DEVICE}')
    print(f'CSV:    {CSV_PATH}')
    print(f'Data:   {DATA_ROOT}')

    # 自适应模型复杂度
    dim, num_blocks, heads = adaptive_model_complexity(NUM_CLASSES)
    print(f'自适应模型配置: dim={dim}, blocks={num_blocks}, heads={heads}')

    # 构建数据集（验证集均衡）
    train_dataset, val_dataset = build_datasets_balanced(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        augment_train=True,
        normalize_method='percentile',
        target_size=(224, 224),
        target_distribution=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    print(f'训练集: {len(train_dataset)} 个样本')
    print(f'验证集: {len(val_dataset)} 个样本')

    # 获取训练集标签用于创建均衡采样器
    train_labels = []
    for i in range(len(train_dataset)):
        _, _, _, label = train_dataset[i]
        train_labels.append(label.item() if hasattr(label, 'item') else label)

    # 创建增强版均衡采样器
    enhanced_sampler = create_enhanced_sampler(train_dataset, train_labels)

    # 数据加载器（使用增强版均衡采样器）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=enhanced_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f'训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}')

    # 模型（自适应复杂度）
    model = DroughtClassifier(
        dim=dim,
        num_blocks=num_blocks,
        heads=heads,
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # 计算增强版类别权重（针对五分类优化）
    df = pd.read_csv(CSV_PATH)
    label_counts = df['label'].value_counts().sort_index()

    # 使用平方根权重，更温和地处理不平衡
    class_weights = 1. / np.sqrt(label_counts.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(DEVICE)

    print('类别权重（平方根处理）:')
    for i, (label, count) in enumerate(label_counts.items()):
        print(f'  标签 {label}: {count}个样本, 权重: {class_weights[i]:.4f}')

    # 损失函数（增强版类别权重和更强的标签平滑）
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # 多阶段优化器
    optimizer_stage1 = Adam(
        model.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    optimizer_stage2 = Adam(
        model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)

    # 多阶段学习率调度器
    scheduler_stage1 = CosineAnnealingLR(optimizer_stage1, T_max=STAGE1_EPOCHS)
    scheduler_stage2 = ReduceLROnPlateau(
        optimizer_stage2, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print('\n开始专家级训练...')

    # 第一阶段：快速收敛
    print(f'\n=== 第一阶段：快速收敛 ({STAGE1_EPOCHS} epochs) ===')
    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()

        print(
            f'\nEpoch {epoch}/{STAGE1_EPOCHS}  (lr={scheduler_stage1.get_last_lr()[0]:.2e})')

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_stage1, DEVICE, stage=1)

        # 验证
        val_loss, val_acc, class_accuracies = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler_stage1.step()

        elapsed = time.time() - t0

        # 打印每个类别的准确率
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%  '
              f'[{elapsed:.1f}s]')

        print('每个类别验证准确率:')
        for i, acc in enumerate(class_accuracies):
            print(f'  标签 {i}: {acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage1.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_accuracies': class_accuracies
            }, BEST_MODEL_PATH)
            print(f'✅ Best model saved! Val Acc: {val_acc:.2f}%')

    # 第二阶段：精细调优
    print(f'\n=== 第二阶段：精细调优 ({STAGE2_EPOCHS} epochs) ===')
    for epoch in range(STAGE1_EPOCHS + 1, STAGE1_EPOCHS + STAGE2_EPOCHS + 1):
        t0 = time.time()

        print(
            f'\nEpoch {epoch}/{EPOCHS}  (lr={optimizer_stage2.param_groups[0]["lr"]:.2e})')

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_stage2, DEVICE, stage=2)

        # 验证
        val_loss, val_acc, class_accuracies = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler_stage2.step(val_acc)

        elapsed = time.time() - t0

        # 打印每个类别的准确率
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%  '
              f'[{elapsed:.1f}s]')

        print('每个类别验证准确率:')
        for i, acc in enumerate(class_accuracies):
            print(f'  标签 {i}: {acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage2.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_accuracies': class_accuracies
            }, BEST_MODEL_PATH)
            print(f'✅ Best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f'\n早停触发！{patience}个epoch验证准确率未提升')
            break

    print(f'\n训练完成。最佳验证准确率: {best_val_acc:.2f}%')
    print(f'最佳模型已保存至: {BEST_MODEL_PATH}')


if __name__ == '__main__':
    main()
