"""
train_ms_exact_55.py
复现之前55%的精确配置
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders

# ============================================================================
# 损失函数
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
# 简单CNN
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1),
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
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# 训练
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for batch in pbar:
        if len(batch) == 4:  # (rgb, tir, ms, labels)
            _, _, ms, labels = batch
        else:  # (ms, labels)
            ms, labels = batch
        
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(ms)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * ms.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += ms.size(0)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    for batch in pbar:
        if len(batch) == 4:  # (rgb, tir, ms, labels)
            _, _, ms, labels = batch
        else:  # (ms, labels)
            ms, labels = batch
        
        ms = ms.to(device)
        logits = model(ms)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        targets_all.extend(labels.numpy())
    
    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average='weighted', zero_division=0)
    
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_exact")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🔄 复现MS 55%精确配置")
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
        modalities=['ms']  # 只用MS
    )
    
    print(f"✅ 数据加载完成")
    
    print("\n🧠 创建模型...")
    model = SimpleCNN(num_classes=5)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    # 计算类别权重
    from datasets.dataset_drought import build_dataloaders
    temp_train_loader, _ = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=256,
        test_size=0.2,
        random_state=42,
        augment_train=False,
        modalities=['ms']
    )
    
    labels_list = []
    for batch in temp_train_loader:
        if len(batch) == 4:
            _, _, _, labels = batch
        else:
            _, labels = batch
        labels_list.extend(labels.numpy())
    
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    print(f"✅ 类别权重: {class_weights.cpu().numpy()}")
    
    # 损失函数 (之前成功的组合)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*80)
    print("🚀 开始训练...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 35
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()
        
        if epoch % 12 == 0 or epoch == 1:
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
            }, os.path.join(args.save_dir, 'ms_exact_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对目标 (55%) 提升: {(best_val_acc - 0.55)*100:+.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()
