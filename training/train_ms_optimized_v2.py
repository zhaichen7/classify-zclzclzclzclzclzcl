"""
train_ms_optimized_v2.py
第2阶段优化: 更强的数据增强 + 过采样 + 类权重调整
预期: 55% → 60-65%
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import DroughtDataset

# ============================================================================
# 1. 高级数据增强类
# ============================================================================

class AdvancedAugmentation:
    """高级遥感图像增强"""
    
    def __init__(self):
        self.p = 0.6
    
    def __call__(self, x):
        """x: (C, H, W) tensor"""
        x = x.clone()
        
        # 随机旋转
        if np.random.rand() < self.p:
            angle = np.random.randint(-45, 46)
            # 使用numpy旋转
            x_np = x.cpu().numpy()
            for c in range(x_np.shape[0]):
                x_np[c] = np.rot90(x_np[c], k=angle // 90)
            x = torch.from_numpy(x_np).float()
        
        # 随机翻转
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[1])  # 水平翻转
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[2])  # 垂直翻转
        
        # 颜色抖动（多光谱数据）
        if np.random.rand() < self.p:
            brightness = np.random.uniform(0.7, 1.3)
            x = x * brightness
            x = torch.clamp(x, 0, 1)
        
        # 随机缩放
        if np.random.rand() < self.p:
            scale = np.random.uniform(0.85, 1.15)
            if scale != 1.0:
                h, w = x.shape[1], x.shape[2]
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 0 and new_w > 0:
                    # 简单的缩放
                    x_scaled = torch.nn.functional.interpolate(
                        x.unsqueeze(0),
                        size=(new_h, new_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
                    # 填充或裁剪到原始大小
                    if new_h > h or new_w > w:
                        x = x_scaled[:, :h, :w]
                    else:
                        x_padded = torch.zeros_like(x)
                        x_padded[:, :new_h, :new_w] = x_scaled
                        x = x_padded
        
        return x

# ============================================================================
# 2. 过采样增强数据集
# ============================================================================

class BalancedDroughtDataset(Dataset):
    """过采样处理不平衡数据的Dataset"""
    
    def __init__(self, base_dataset, oversample_ratio=2.0, augmentation=None):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        
        # 统计类别分布
        self.class_indices = [[] for _ in range(5)]
        for idx in range(len(base_dataset)):
            try:
                _, _, _, label = base_dataset[idx]
                self.class_indices[label].append(idx)
            except:
                pass
        
        # 找最多的类
        max_class_size = max(len(indices) for indices in self.class_indices)
        
        # 过采样少数类
        self.balanced_indices = []
        for class_idx, indices in enumerate(self.class_indices):
            if len(indices) > 0:
                target_size = int(max_class_size * oversample_ratio)
                oversampled = np.random.choice(indices, size=target_size, replace=True)
                self.balanced_indices.extend(oversampled)
        
        np.random.shuffle(self.balanced_indices)
        print(f"✅ 过采样完成: {len(self.balanced_indices)} 个样本 ({oversample_ratio}x)")
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def __getitem__(self, idx):
        base_idx = self.balanced_indices[idx]
        rgb, tir, ms, label = self.base_dataset[base_idx]
        
        # 应用增强
        if self.augmentation is not None and np.random.rand() < 0.6:
            ms = self.augmentation(ms)
        
        return rgb, tir, ms, label

# ============================================================================
# 3. Focal Loss
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
# 4. 训练函数
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
# 5. 主训练函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MS单模态优化训练 - 阶段2")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--label_smoothing", type=float, default=0.15)
    parser.add_argument("--oversample_ratio", type=float, default=2.0)
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
    print(f"过采样倍数: {args.oversample_ratio}x")
    print("="*70)
    
    # 加载数据集
    print("\n📊 加载数据...")
    df = pd.read_csv(args.csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_ds = DroughtDataset(train_df, data_root=args.data_root, modalities=['ms'])
    val_ds = DroughtDataset(val_df, data_root=args.data_root, modalities=['ms'])
    
    # 应用过采样和增强
    augmentation = AdvancedAugmentation()
    train_ds_balanced = BalancedDroughtDataset(
        train_ds,
        oversample_ratio=args.oversample_ratio,
        augmentation=augmentation
    )
    
    train_loader = DataLoader(
        train_ds_balanced,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✅ 验证集大小: {len(val_ds)}")
    
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
                nn.Dropout(0.4),
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
    for idx in range(len(train_ds_balanced)):
        try:
            _, _, _, label = train_ds_balanced[idx]
            train_labels.append(label)
        except:
            pass
    
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
        return 0.6 * focal + 0.4 * smooth
    
    # 优化器和学习率调度
    print("\n⚙️  创建优化器...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    best_class_accs = None
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
