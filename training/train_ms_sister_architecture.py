"""
train_ms_sister_architecture.py
借鉴师姐的双分支架构 + 强分类头
适配MS五分类任务
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
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_drought import build_dataloaders

# ============================================================================
# 借鉴师姐的双分支设计
# ============================================================================

class MSFeatureBranch(nn.Module):
    """单个特征提取分支 (参考师姐的设计)"""
    def __init__(self, in_channels=8):
        super().__init__()
        
        self.features = nn.Sequential(
            # 第1块
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 112x112
            
            # 第2块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 56x56
            
            # 第3块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 28x28
            
            # 第4块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )
    
    def forward(self, x):
        return self.features(x)

class DualBranchMSClassifier(nn.Module):
    """
    双分支MS分类器 - 完全参考师姐的架构
    不同之处：
      - 两个分支输入都是MS (而师姐是control和experimental)
      - 但思想完全一致：多角度特征提取 → 融合 → 强分类头
    """
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 两个独立的特征提取分支
        # 分支1：直接提取MS特征
        self.branch1 = MSFeatureBranch(in_channels=8)
        
        # 分支2：也提取MS特征 (用不同的卷积核/初始化)
        # 模拟师姐的两个不同分支思想
        self.branch2 = MSFeatureBranch(in_channels=8)
        
        # 动态计算特征维度 (参考师姐的做法)
        with torch.no_grad():
            dummy = torch.randn(1, 8, 224, 224)
            feat1 = self.branch1(dummy)
            feat1_flat = feat1.view(feat1.size(0), -1)
            self.branch1_dim = feat1_flat.shape[1]
            
            feat2 = self.branch2(dummy)
            feat2_flat = feat2.view(feat2.size(0), -1)
            self.branch2_dim = feat2_flat.shape[1]
        
        print(f"Branch1 dim: {self.branch1_dim}, Branch2 dim: {self.branch2_dim}")
        
        # 师姐风格的强分类头
        total_dim = self.branch1_dim + self.branch2_dim
        
        self.classifier = nn.Sequential(
            # 第1层
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            # 第2层
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            # 第3层
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # 第4层
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 输出层
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # 两个分支提取特征
        feat1 = self.branch1(x)
        feat1 = feat1.view(feat1.size(0), -1)
        
        feat2 = self.branch2(x)
        feat2 = feat2.view(feat2.size(0), -1)
        
        # 融合特征
        combined = torch.cat([feat1, feat2], dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
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
        logits = model(ms)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
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
        
        logits = model(ms)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.2f}%'})
    
    class_accs = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(5)]
    
    return total_loss / total, correct / total, class_accs

def main():
    parser = argparse.ArgumentParser(description="MS单模态 - 师姐架构版本")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_sister_arch")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("🚀 MS单模态 - 师姐架构版本 (双分支 + 强分类头)")
    print("="*70)
    
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
    
    print("\n🧠 创建模型...")
    model = DualBranchMSClassifier(num_classes=5)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    # 计算类别权重
    print("\n⚖️  计算类别权重...")
    labels_list = []
    for _, _, _, labels in train_loader:
        labels_list.extend(labels.numpy())
    
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    print(f"✅ 类别权重: {class_weights.cpu().numpy()}")
    
    # 损失函数
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 25
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
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
            }, os.path.join(args.save_dir, 'drought_best.pth'))
            
            print(f"           ✅ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发 (patience={patience})")
                break
    
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对基础 (57.5%) 提升: {(best_val_acc - 0.575)*100:+.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("="*70)

if __name__ == '__main__':
    main()
