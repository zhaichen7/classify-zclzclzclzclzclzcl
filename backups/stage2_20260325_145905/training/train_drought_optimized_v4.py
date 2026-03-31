"""
优化版训练脚本 v4
- 验证集均衡 ✅
- 训练集均衡 ✅
- 类别权重 ✅
- 每类准确率 ✅
- 模型简化 ✅
- 训练稳定化 ✅
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import pandas as pd
import numpy as np

from net_drought import DroughtClassifier
from dataset_drought import build_datasets_balanced

# ========== 优化配置 ==========
CSV_PATH = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'

# 训练参数优化
BATCH_SIZE = 4           # 保持批次大小4，避免显存不足
EPOCHS = 50
LR = 5e-5               # 降低学习率，更稳定
NUM_CLASSES = 5
NUM_WORKERS = 4
RANDOM_SEED = 123
TEST_SIZE = 0.2

# 正则化参数
WEIGHT_DECAY = 1e-3     # 增强权重衰减
LABEL_SMOOTHING = 0.1   # 标签平滑
GRAD_CLIP = 1.0         # 梯度裁剪

AUGMENTATION_FACTOR = 5

SAVE_DIR = './models_optimized'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_best_optimized.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_balanced_sampler(dataset, labels):
    """创建均衡采样器"""
    # 计算每个类别的样本权重（少数类别权重更高）
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]

    # 创建加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch（带梯度裁剪）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
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

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

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
def evaluate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # 每个类别的准确率统计
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

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

    return epoch_loss, epoch_acc, class_accuracies


def main():
    print('=' * 60)
    print('优化版干旱分级训练 v4')
    print('=' * 60)

    print(f'Device: {DEVICE}')
    print(f'CSV:    {CSV_PATH}')
    print(f'Data:   {DATA_ROOT}')
    print(f'增强倍数: {AUGMENTATION_FACTOR}')

    # 打印优化配置
    print('\n优化配置:')
    print(f'  模型大小: dim=32 (简化模型)')
    print(f'  网络层数: [2,2] (减少过拟合)')
    print(f'  学习率: {LR} (更稳定)')
    print(f'  Weight Decay: {WEIGHT_DECAY} (强正则化)')
    print(f'  Batch Size: {BATCH_SIZE} (避免显存不足)')
    print(f'  Label Smoothing: {LABEL_SMOOTHING} (减少过拟合)')
    print(f'  梯度裁剪: {GRAD_CLIP} (稳定训练)')

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

    # 创建均衡采样器
    balanced_sampler = create_balanced_sampler(train_dataset, train_labels)

    # 数据加载器（使用均衡采样器）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=balanced_sampler,
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

    # 模型（简化模型复杂度）
    model = DroughtClassifier(
        dim=32,           # 减小模型维度
        num_blocks=[2, 2],  # 减少网络层数
        heads=[4, 4, 4],  # 减少注意力头数
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # 计算类别权重（用于损失函数）
    df = pd.read_csv(CSV_PATH)
    label_counts = df['label'].value_counts().sort_index()
    class_weights = 1. / torch.tensor(label_counts.values, dtype=torch.float)
    class_weights = class_weights.to(DEVICE)

    # 损失函数（带类别权重和标签平滑）
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    print('\n开始训练...')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        print(
            f'\nEpoch {epoch}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})')

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_loss, val_acc, class_accuracies = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()

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
                'optimizer_state_dict': optimizer.state_dict(),
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
