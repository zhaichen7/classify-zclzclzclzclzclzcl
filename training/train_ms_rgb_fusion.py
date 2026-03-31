"""
train_ms_rgb_fusion.py
RGB + MS 高级融合：注意力机制动态融合两种模态
预期: 55% → 65-75%

核心思想:
  不是简单拼接，而是用注意力学习"什么时候信任RGB，什么时候信任MS"
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
# 双路编码器 + 注意力融合
# ============================================================================

class DualPathFusionNet(nn.Module):
    """
    双路融合网络：
    RGB encoder → 特征表示
                 ↘
                  注意力融合 → 分类
                 ↗
    MS encoder → 特征表示
    """
    def __init__(self, num_classes=5):
        super().__init__()
        
        # ========== RGB 编码器 ==========
        # 轻量 CNN 用于 RGB (3通道)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== MS 编码器 ==========
        # 轻量 CNN 用于 MS (8通道)
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== 注意力融合层 ==========
        # 输入: RGB特征 (256) + MS特征 (256)
        # 输出: 融合权重 (2个，代表RGB和MS的重要性)
        self.attention = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)  # 确保两个权重和为1
        )
        
        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, rgb, ms):
        """
        rgb: (B, 3, 224, 224)
        ms:  (B, 8, 224, 224)
        """
        # 编码
        rgb_feat = self.rgb_encoder(rgb)  # (B, 256, 1, 1)
        ms_feat = self.ms_encoder(ms)     # (B, 256, 1, 1)
        
        # 展平
        rgb_feat_flat = rgb_feat.view(rgb_feat.size(0), -1)  # (B, 256)
        ms_feat_flat = ms_feat.view(ms_feat.size(0), -1)     # (B, 256)
        
        # 注意力融合
        concat_feat = torch.cat([rgb_feat_flat, ms_feat_flat], dim=1)  # (B, 512)
        attn_weights = self.attention(concat_feat)  # (B, 2)
        
        # attn_weights[:, 0] 是 RGB 的权重
        # attn_weights[:, 1] 是 MS 的权重
        
        # 加权融合
        rgb_weight = attn_weights[:, 0].view(-1, 1)  # (B, 1)
        ms_weight = attn_weights[:, 1].view(-1, 1)   # (B, 1)
        
        fused_feat = rgb_weight * rgb_feat_flat + ms_weight * ms_feat_flat  # (B, 256)
        
        # 分类
        logits = self.classifier(fused_feat)  # (B, 5)
        
        return logits, attn_weights

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
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, attn_weights = model(rgb, ms)
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
    rgb_weights_list = []
    ms_weights_list = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        logits, attn_weights = model(rgb, ms)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 记录注意力权重
        rgb_weights_list.append(attn_weights[:, 0].cpu().numpy())
        ms_weights_list.append(attn_weights[:, 1].cpu().numpy())
        
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
    
    # 计算平均注意力权重
    rgb_weight_mean = np.concatenate(rgb_weights_list).mean()
    ms_weight_mean = np.concatenate(ms_weights_list).mean()
    
    return epoch_loss, epoch_acc, class_accs, rgb_weight_mean, ms_weight_mean

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RGB+MS 融合分类")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_fusion")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("🔗 RGB + MS 融合分类 (注意力机制)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    # 加载数据（需要 RGB + MS）
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
        modalities=['rgb', 'ms']
    )
    
    print(f"✅ 数据加载完成")
    
    # 创建模型
    print("\n🧠 创建融合网络...")
    model = DualPathFusionNet(num_classes=5)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 总参数: {total_params:,}")
    print(f"✅ 可训练参数: {trainable_params:,}")
    
    # 计算类别权重
    print("\n⚖️  计算类别权重...")
    from collections import Counter
    train_labels = []
    for rgb, _, ms, labels in train_loader:
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
    
    # 优化器
    print("\n⚙️  创建优化器...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练循环
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs, rgb_w, ms_w = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"\n✓ Epoch {epoch:3d}/{args.epochs} | LR={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        
        class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
        print(f"           Per-class: {class_str}")
        print(f"           注意力权重: RGB={rgb_w:.3f}, MS={ms_w:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'fusion_best.pth'))
            
            print(f"           ✅ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发 (patience={patience})")
                break
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ RGB+MS 融合训练完成！")
    print("="*70)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对第1阶段提升: {(best_val_acc - 0.55)*100:+.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'fusion_best.pth')}")
    print("="*70)

if __name__ == '__main__':
    main()
