"""
train_ms_optimized_v2.py - 修复版本
第2阶段优化: 更强的数据增强 + 更多轮数
预期: 55% → 62-68%
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders

# ============================================================================
# 1. Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# ============================================================================
# 2. 训练函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for batch_idx, (rgb, tir, ms, labels) in enumerate(pbar):
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for rgb, tir, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })
    
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    class_accs = []
    for i in range(5):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)
    
    return epoch_loss, epoch_acc, class_accs

# ============================================================================
# 3. 主训练函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MS单模态优化训练 - 阶段2")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)  # 更多轮数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--label_smoothing", type=float, default=0.2)  # 更强
    parser.add_argument("--save_dir", default="./models_ms_opt_v2")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("🚀 MS单模态优化训练 - 第2阶段")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Focal Loss Gamma: {args.gamma}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print("="*70)
    
    # 加载数据 - 启用数据增强
    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,  # 启用增强
        balanced=True,
        modalities=['ms']
    )
    
    print(f"✅ 数据加载完成")
    
    # 创建模型
    print("\n🧠 创建模型...")
    encoder = RestormerEncoder(
        inp_channels=8, dim=48, num_blocks=[4, 6],
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias'
    )
    
    class MSClassifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(48, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),  # 增加 dropout
                nn.Linear(128, 5)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.pool(x).view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = MSClassifier(encoder)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数数: {total_params:,}")
    
    # 计算类别权重
    print("\n⚖️  计算类别权重...")
    from collections import Counter
    train_labels = []
    for _, _, _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for i in range(5):
        count = label_counts.get(i, 1)
        weight = total_samples / (5 * count)
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"✅ 类别权重: {[f'{w:.3f}' for w in class_weights.cpu().numpy()]}")
    
    # 损失函数
    print("\n🎯 创建损失函数...")
    focal_loss = FocalLoss(alpha=class_weights, gamma=args.gamma)
    label_smooth_loss = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
        weight=class_weights
    )
    
    def criterion(outputs, targets):
        focal = focal_loss(outputs, targets)
        smooth = label_smooth_loss(outputs, targets)
        return 0.7 * focal + 0.3 * smooth  # 增加focal比例
    
    # 优化器和学习率调度
    print("\n⚙️  创建优化器...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 更激进的学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )
    
    # 训练循环
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    best_class_accs = None
    patience = 30  # 增加耐心
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"\n✓ Epoch {epoch:3d}/{args.epochs} | LR={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        
        class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
        print(f"           Per-class acc: {class_str}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_class_accs = class_accs
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
            }, os.path.join(args.save_dir, 'drought_best.pth'))
            
            print(f"           ✅ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发 (patience={patience})")
                break
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ 第2阶段训练完成！")
    print("="*70)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"类别准确率: {' | '.join([f'c{i}={best_class_accs[i]*100:.1f}%' for i in range(5)])}")
    print(f"模型保存: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("="*70)

if __name__ == '__main__':
    main()
