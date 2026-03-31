"""
train_multiarch_focal.py
多模态干旱分级 + Focal Loss
用于处理类别严重不平衡（类别0只有37个样本）
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought import DroughtClassifier
from datasets.dataset_drought import DroughtDataset, build_datasets
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不平衡问题
    Reference: Lin et al., ICCV 2017 (Focal Loss for Dense Object Detection)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重，shape: (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, num_classes), targets: (B,)
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        
        # 计算预测概率
        p = torch.exp(-ce_loss)
        
        # Focal Loss: loss = -alpha * (1 - p)^gamma * log(p)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device, non_blocking=True)
        tir = tir.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.1f}%'
        })

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    pbar = tqdm(loader, desc="Validation", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device, non_blocking=True)
        tir = tir.to(device, non_blocking=True)
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        # 统计每个类别
        for i in range(num_classes):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    class_accs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)

    return epoch_loss, epoch_acc, class_accs


def main():
    parser = argparse.ArgumentParser(description="多模态干旱分级 + Focal Loss")
    parser.add_argument("--csv_path", type=str, required=True, help="标签CSV路径")
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    
    # Focal Loss 参数
    parser.add_argument("--use_focal_loss", type=bool, default=True, help="使用Focal Loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma参数")
    
    # 类别权重
    parser.add_argument("--class_weights", type=float, nargs=5, default=[5.0, 1.0, 1.0, 1.0, 1.0], 
                       help="每个类别的权重（类别0优先）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./models_focal")
    parser.add_argument("--dim", type=int, default=48)
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("多模态干旱分级 + Focal Loss 训练")
    print("="*70)
    print(f"Device: {device}")
    print(f"Focal Loss: gamma={args.focal_gamma}")
    print(f"类别权重: {args.class_weights}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("="*70)
    
    # 加载数据（禁用增强以避免模块问题）
    print("\n加载数据...")
    train_ds, val_ds = build_datasets(
        csv_path=args.csv_path,
        data_root=args.data_root,
        test_size=0.2,
        random_state=42,
        augment_train=False,  # 禁用增强
        normalize_method='percentile',
        target_size=(224, 224),
        balanced=True,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    print(f"✅ 训练集: {len(train_ds)}, 验证集: {len(val_ds)}")
    
    # 创建模型
    print("\n创建模型...")
    model = DroughtClassifier(
        dim=args.dim,
        num_blocks=[4, 4],
        heads=[8, 8, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        num_classes=5
    ).to(device)
    
    print(f"✅ 模型参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 类别权重
    class_weights = torch.tensor(args.class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # 损失函数
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"✅ 使用 Focal Loss（gamma={args.focal_gamma}）")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("✅ 使用加权交叉熵损失")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_val_acc = 0
    best_epoch = 0
    best_class_accs = None
    best_model_path = os.path.join(args.save_dir, "drought_best_focal.pth")
    
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        
        # 打印每个类别的准确率
        class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
        print(f"          Per-class acc: {class_str}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_class_accs = class_accs
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'class_accs': class_accs
            }, best_model_path)
            print(f"          ✅ 保存最佳模型 (val_acc={val_acc*100:.2f}%)")
        
        scheduler.step()
    
    # 最终统计
    print("\n" + "="*70)
    print("训练完成 — 性能汇总")
    print("="*70)
    print(f"最佳 epoch: {best_epoch}")
    print(f"最佳 val_acc: {best_val_acc*100:.2f}%")
    print(f"最佳 per-class acc: {' | '.join([f'c{i}={best_class_accs[i]*100:.1f}%' for i in range(5)])}")
    print(f"模型保存至: {best_model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
