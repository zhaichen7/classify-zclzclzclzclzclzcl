"""
train_trimodal_kfold_fixed.py
K折交叉验证 + 三模态融合 (RGB + TIR + MS)
预期: 55% → 70-80%
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 简单数据集加载
# ============================================================================

class SimpleTrimodalDataset(Dataset):
    """简单三模态数据集"""
    def __init__(self, df, data_root='dataset/'):
        self.df = df
        self.data_root = data_root
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        sample_id = row['sample_id']
        
        # 加载三个模态
        rgb_path = os.path.join(self.data_root, f"RGB_{sample_id}.npy")
        tir_path = os.path.join(self.data_root, f"TIR_{sample_id}.npy")
        ms_path = os.path.join(self.data_root, f"MS_{sample_id}.npy")
        
        # 加载数据
        rgb = np.load(rgb_path).astype(np.float32)
        tir = np.load(tir_path).astype(np.float32)
        ms = np.load(ms_path).astype(np.float32)
        
        # 归一化
        rgb = rgb / 255.0 if rgb.max() > 1 else rgb
        tir = tir / 255.0 if tir.max() > 1 else tir
        ms = ms / 255.0 if ms.max() > 1 else ms
        
        return torch.from_numpy(rgb), torch.from_numpy(tir), torch.from_numpy(ms), label

# ============================================================================
# 三模态融合网络
# ============================================================================

class TrimodalFusionNet(nn.Module):
    """RGB + TIR + MS 三模态融合"""
    def __init__(self, num_classes=5):
        super().__init__()
        
        # RGB 编码器 (3通道)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # TIR 编码器 (1通道)
        self.tir_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MS 编码器 (8通道)
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(8, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 融合权重学习
        self.fusion_weights = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, tir, ms):
        # 编码
        rgb_feat = self.rgb_encoder(rgb).view(rgb.size(0), -1)  # (B, 256)
        tir_feat = self.tir_encoder(tir).view(tir.size(0), -1)  # (B, 256)
        ms_feat = self.ms_encoder(ms).view(ms.size(0), -1)      # (B, 256)
        
        # 学习融合权重
        concat_feats = torch.cat([rgb_feat, tir_feat, ms_feat], dim=1)  # (B, 768)
        weights = self.fusion_weights(concat_feats)  # (B, 3)
        
        # 加权融合
        fused = (weights[:, 0:1] * rgb_feat + 
                 weights[:, 1:2] * tir_feat + 
                 weights[:, 2:3] * ms_feat)  # (B, 256)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits, weights

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

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for rgb, tir, ms, labels in train_loader:
        rgb, tir, ms, labels = rgb.to(device), tir.to(device), ms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(rgb, tir, ms)
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
def evaluate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for rgb, tir, ms, labels in val_loader:
        rgb, tir, ms, labels = rgb.to(device), tir.to(device), ms.to(device), labels.to(device)
        
        logits, _ = model(rgb, tir, ms)
        preds = logits.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
    
    acc = correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return acc, f1

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_trimodal_kfold")
    parser.add_argument("--num_folds", type=int, default=5)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print(f"🔗 三模态融合 K折交叉验证 (K={args.num_folds})")
    print("="*80)
    
    # 加载数据
    print("\n📊 加载数据集...")
    df = pd.read_csv(args.csv_path)
    dataset = SimpleTrimodalDataset(df, data_root=args.data_root)
    labels = df['label'].values
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    # K折划分
    print(f"\n🔄 进行 {args.num_folds} 折交叉验证...\n")
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_fold_preds = []
    all_fold_targets = np.array([])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*80}")
        print(f"Fold {fold+1}/{args.num_folds} | 训练: {len(train_idx)}, 验证: {len(val_idx)}")
        print(f"{'='*80}")
        
        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # 创建模型
        model = TrimodalFusionNet(num_classes=5)
        model.to(device)
        
        # 计算类别权重
        from collections import Counter
        train_labels = labels[train_idx]
        label_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor(
            [total / (5 * label_counts.get(i, 1)) for i in range(5)],
            dtype=torch.float, device=device
        )
        class_weights = class_weights / class_weights.sum() * 5
        
        # 损失函数
        focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        
        def criterion(outputs, targets):
            return 0.6 * focal_loss(outputs, targets) + 0.4 * ce_loss(outputs, targets)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 训练循环
        best_acc = 0
        best_f1 = 0
        patience = 20
        patience_counter = 0
        best_preds = None
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            if epoch % 30 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: train={train_acc*100:.1f}%, val={val_acc*100:.1f}%, f1={val_f1*100:.1f}%")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                patience_counter = 0
                
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'fold{fold+1}_best.pth'))
                
                # 保存最佳预测
                model.eval()
                with torch.no_grad():
                    preds = []
                    for rgb, tir, ms, _ in val_loader:
                        rgb, tir, ms = rgb.to(device), tir.to(device), ms.to(device)
                        logits, _ = model(rgb, tir, ms)
                        preds.extend(logits.argmax(dim=1).cpu().numpy())
                    best_preds = np.array(preds)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发 (Epoch {epoch})")
                    break
        
        fold_results.append({'fold': fold+1, 'acc': best_acc, 'f1': best_f1})
        all_fold_preds.append(best_preds)
        all_fold_targets = np.concatenate([all_fold_targets, labels[val_idx]])
        
        print(f"✅ Fold {fold+1}: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%")
    
    # 集成结果
    print(f"\n{'='*80}")
    print("🎯 K折集成结果")
    print(f"{'='*80}")
    
    ensemble_preds = np.concatenate(all_fold_preds)
    ensemble_acc = accuracy_score(all_fold_targets, ensemble_preds)
    ensemble_f1 = f1_score(all_fold_targets, ensemble_preds, average='weighted', zero_division=0)
    
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    
    print(f"\n单模型平均:")
    print(f"  准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    
    print(f"\n集成模型:")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    print(f"\n相对第1阶段 (55%) 提升:")
    print(f"  单模型: {(np.mean(accs)-0.55)*100:+.2f}%")
    print(f"  集成模型: {(ensemble_acc-0.55)*100:+.2f}%")
    
    print(f"\n各折结果:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['acc']*100:.2f}%")
    
    print(f"\n✅ 完成！模型已保存到 {args.save_dir}/")
    print("="*80)

if __name__ == '__main__':
    main()
