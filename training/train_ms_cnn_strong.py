"""
train_ms_cnn_strong.py
MS单模态 - 强化CNN方案
目标: 55% → 65%+

策略:
  1. ResNet50 作为骨干网络 (已被验证有效)
  2. 更强的数据增强 (Augmentation)
  3. 更激进的正则化 (Dropout + 权重衰减)
  4. 多损失函数组合
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
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 数据加载
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, csv_path, data_root, ids, augment=False):
        from datasets.dataset_drought import get_file_paths, read_envi_band, percentile_normalize, compute_ndvi, compute_gndvi, compute_savi
        
        df = pd.read_csv(csv_path)
        df = df[df['id'].isin(ids)].reset_index(drop=True)
        self.ids = df['id'].tolist()
        self.labels = df['label'].tolist()
        
        self.data_root = data_root
        self.augment = augment
        self.get_file_paths = get_file_paths
        self.read_envi_band = read_envi_band
        self.normalize = percentile_normalize
        self.compute_ndvi = compute_ndvi
        self.compute_gndvi = compute_gndvi
        self.compute_savi = compute_savi
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[idx]
        paths = self.get_file_paths(sample_id, self.data_root)
        
        nir_arr = self.read_envi_band(paths['nir'], band_idx=0)
        red_arr = self.read_envi_band(paths['red'], band_idx=0)
        green_arr = self.read_envi_band(paths['green'], band_idx=0)
        blue_arr = self.read_envi_band(paths['blue'], band_idx=0)
        rededge_arr = self.read_envi_band(paths['rededge'], band_idx=0)
        
        ndvi = self.compute_ndvi(nir_arr, red_arr)
        gndvi = self.compute_gndvi(nir_arr, green_arr)
        savi = self.compute_savi(nir_arr, red_arr)
        
        ms = np.stack([nir_arr, red_arr, blue_arr, green_arr, rededge_arr, ndvi, gndvi, savi], axis=0)
        ms = self.normalize(ms)
        
        ms = np.stack([cv2.resize(ms[i], (224, 224), interpolation=cv2.INTER_LINEAR) for i in range(8)], axis=0)
        
        # 强数据增强
        if self.augment:
            # 随机旋转
            if np.random.rand() > 0.5:
                angle = np.random.randint(-20, 20)
                h, w = ms.shape[1], ms.shape[2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                ms = np.stack([cv2.warpAffine(ms[i], M, (w, h)) for i in range(8)], axis=0)
            
            # 随机缩放
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.8, 1.2)
                h, w = int(ms.shape[1] * scale), int(ms.shape[2] * scale)
                ms = np.stack([cv2.resize(ms[i], (224, 224)) for i in range(8)], axis=0)
            
            # 随机噪声
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 0.02, ms.shape)
                ms = np.clip(ms + noise, 0, 1)
            
            # 翻转
            if np.random.rand() > 0.5:
                ms = np.flip(ms, axis=-1).copy()
            if np.random.rand() > 0.5:
                ms = np.flip(ms, axis=-2).copy()
        
        return torch.from_numpy(ms).float(), torch.tensor(label, dtype=torch.long)

# ============================================================================
# ResNet50 + 自定义头
# ============================================================================

class MSResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 加载预训练 ResNet50 (RGB用)
        resnet = models.resnet50(weights=None)
        
        # 修改第一层以支持8通道输入
        self.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.avgpool = resnet.avgpool
        
        # 自定义分类头 (更强的正则化)
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

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
# 训练
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for ms, labels in pbar:
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
    for ms, labels in pbar:
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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_dir", default="./models_ms_resnet50_strong")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🚀 MS单模态 - ResNet50强化方案")
    print("="*80)
    
    print("\n📂 加载数据...")
    df = pd.read_csv(args.csv_path)
    train_ids, val_ids = train_test_split(df['id'].tolist(), test_size=0.2, random_state=42)
    
    train_ds = SimpleDataset(args.csv_path, args.data_root, train_ids, augment=True)
    val_ds = SimpleDataset(args.csv_path, args.data_root, val_ids, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"✅ 训练集: {len(train_ds)}, 验证集: {len(val_ds)}")
    
    print("\n🧠 创建模型...")
    model = MSResNet50(num_classes=5)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    # 损失函数
    label_counts = Counter([train_ds[i][1].item() for i in range(len(train_ds))])
    total_samples = len(train_ds)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.15, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    print("\n" + "="*80)
    print("🚀 开始训练...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step()
        
        if epoch % 15 == 0 or epoch == 1:
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
            }, os.path.join(args.save_dir, 'ms_resnet50_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ MS单模态ResNet50训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对之前 (55%) 提升: {(best_val_acc - 0.55)*100:+.2f}%")
    if best_val_acc >= 0.65:
        print("✅ 达到目标 65%+")
    print("="*80)

if __name__ == '__main__':
    main()
