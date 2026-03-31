"""
train_ms_rgb_fusion_v2.py
RGB + MS 高级融合 v2：多头通道注意力 + 空间注意力 + 特征金字塔
预期: 55% → 70-80%
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders

# ============================================================================
# 注意力模块
# ============================================================================

class ChannelAttention(nn.Module):
    """多头通道注意力"""
    def __init__(self, in_channels, reduction=16, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        out = self.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    """通道-空间注意力块"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ============================================================================
# 特征金字塔融合
# ============================================================================

class FeaturePyramidFusion(nn.Module):
    """特征金字塔融合"""
    def __init__(self, in_channels_rgb, in_channels_ms, out_channels):
        super().__init__()
        
        self.rgb_adapt = nn.Sequential(
            nn.Conv2d(in_channels_rgb, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.ms_adapt = nn.Sequential(
            nn.Conv2d(in_channels_ms, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels)
        )
    
    def forward(self, rgb_feat, ms_feat):
        rgb_adapted = self.rgb_adapt(rgb_feat)
        ms_adapted = self.ms_adapt(ms_feat)
        
        if rgb_adapted.size(2) != ms_adapted.size(2):
            size = (rgb_adapted.size(2), rgb_adapted.size(3))
            ms_adapted = nn.functional.interpolate(
                ms_adapted, size=size, mode='bilinear', align_corners=False
            )
        
        fused = torch.cat([rgb_adapted, ms_adapted], dim=1)
        fused = self.fusion(fused)
        
        return fused

# ============================================================================
# 双路编码器 + 高级融合
# ============================================================================

class AdvancedFusionNet(nn.Module):
    """RGB + MS 高级融合网络"""
    def __init__(self, num_classes=5):
        super().__init__()
        
        # RGB 编码器
        self.rgb_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32),
            nn.MaxPool2d(2)
        )
        
        self.rgb_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64)
        )
        
        self.rgb_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128)
        )
        
        self.rgb_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        
        # MS 编码器
        self.ms_layer1 = nn.Sequential(
            nn.Conv2d(8, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32),
            nn.MaxPool2d(2)
        )
        
        self.ms_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64)
        )
        
        self.ms_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128)
        )
        
        self.ms_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256)
        )
        
        # 特征金字塔融合
        self.pyramid_fusion2 = FeaturePyramidFusion(64, 64, 64)
        self.pyramid_fusion3 = FeaturePyramidFusion(128, 128, 128)
        self.pyramid_fusion4 = FeaturePyramidFusion(256, 256, 256)
        
        # 分类头
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, ms):
        # RGB 层级特征
        rgb_l1 = self.rgb_layer1(rgb)
        rgb_l2 = self.rgb_layer2(rgb_l1)
        rgb_l3 = self.rgb_layer3(rgb_l2)
        rgb_l4 = self.rgb_layer4(rgb_l3)
        
        # MS 层级特征
        ms_l1 = self.ms_layer1(ms)
        ms_l2 = self.ms_layer2(ms_l1)
        ms_l3 = self.ms_layer3(ms_l2)
        ms_l4 = self.ms_layer4(ms_l3)
        
        # 多层级融合
        fused_l2 = self.pyramid_fusion2(rgb_l2, ms_l2)
        fused_l3 = self.pyramid_fusion3(rgb_l3, ms_l3)
        fused_l4 = self.pyramid_fusion4(rgb_l4, ms_l4)
        
        # 分类
        final_feat = self.global_avgpool(fused_l4)
        final_feat_flat = final_feat.view(final_feat.size(0), -1)
        logits = self.classifier(final_feat_flat)
        
        return logits

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

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(rgb, ms)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        
        logits = model(rgb, ms)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        targets_all.extend(labels.numpy())
    
    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average='weighted', zero_division=0)
    
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_fusion_v2")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🔗 RGB + MS 高级融合 v2 (CBAM + 特征金字塔)")
    print("="*80)
    
    print("\n📂 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['rgb', 'tir', 'ms']
    )
    print(f"✅ 数据加载完成")
    
    print("\n🧠 创建模型...")
    model = AdvancedFusionNet(num_classes=5)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
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
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*80)
    print("🚀 开始训练...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 40
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
                  f"val_acc={val_acc*100:.2f}%, val_f1={val_f1*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, os.path.join(args.save_dir, 'fusion_v2_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ RGB+MS 高级融合训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对第1阶段 (55%) 提升: {(best_val_acc - 0.55)*100:+.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'fusion_v2_best.pth')}")
    print("="*80)

if __name__ == '__main__':
    main()
