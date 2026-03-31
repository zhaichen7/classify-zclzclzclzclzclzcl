"""
train_ms_resnet50_fixed.py
ResNet50 预训练 - 改进版 (解决 BN + 小batch 问题)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders

# ============================================================================
# ResNet50 预训练分类器 - 改进版
# ============================================================================

class ResNet50MSClassifier(nn.Module):
    """ResNet50 预训练 + MS 多光谱分类 (改进版)"""
    def __init__(self, num_classes=5, in_channels=8, pretrained=True):
        super().__init__()
        
        # 加载预训练 ResNet50
        import torchvision.models as models
        try:
            resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        except:
            resnet50 = models.resnet50(pretrained=True)
        
        # 修改第一层卷积: RGB(3通道) → MS(8通道)
        original_conv1 = resnet50.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet50.bn1
        
        # 初始化权重
        with torch.no_grad():
            rgb_weight = original_conv1.weight  # (64, 3, 7, 7)
            avg_weight = rgb_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            self.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
        
        # ✅ 冻结 BN 层的参数更新（解决小batch BN不稳定问题）
        # 这样 BN 只用预训练的统计量，不会被小batch污染
        self.bn1.eval()
        for p in self.bn1.parameters():
            p.requires_grad = False
        
        # 冻结早期层
        for param in resnet50.layer1.parameters():
            param.requires_grad = False
        
        # ✅ 冻结 layer2 的 BN（小batch 影响很大）
        for module in resnet50.layer2.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        
        # 构建特征提取器
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool
        )
        
        # ✅ 分类头：去掉 BatchNorm1d，用 LayerNorm 替代（对小batch友好）
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),  # 用 LayerNorm 替代 BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 用 LayerNorm 替代 BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def train(self, mode=True):
        """覆盖 train 方法，确保 backbone 中所有 BN 始终保持 eval 模式"""
        super().train(mode)
        for module in self.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
        return self

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# 训练函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.2f}%'})
    
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
    for _, _, ms, labels in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.2f}%'})
    
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
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ResNet50 MS 单模态训练 (改进版)")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_resnet50")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("🚀 ResNet50 预训练 - MS 单模态训练 (改进版)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("\n✅ 改进点:")
    print("  • 冻结 BN 层的参数（防止小batch污染）")
    print("  • 分类头用 LayerNorm 替代 BatchNorm1d（对小batch友好）")
    print("="*70)
    
    # 加载数据
    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['ms']
    )
    
    print(f"✅ 数据加载完成")
    
    # 创建模型
    print("\n🧠 创建 ResNet50 模型...")
    model = ResNet50MSClassifier(num_classes=5, in_channels=8, pretrained=True)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数数: {total_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
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
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    print(f"✅ 损失函数: Focal Loss (0.6) + Label Smoothing (0.4)")
    
    # 优化器
    print("\n⚙️  创建优化器...")
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    print(f"✅ 优化器: AdamW (lr={args.lr})")
    print(f"✅ 学习率调度: CosineAnnealing")
    
    # 训练循环
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    best_class_accs = None
    patience = 40
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
    print("✅ ResNet50 训练完成！")
    print("="*70)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"类别准确率: {' | '.join([f'c{i}={best_class_accs[i]*100:.1f}%' for i in range(5)])}")
    print(f"模型保存: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("="*70)

if __name__ == '__main__':
    main()
