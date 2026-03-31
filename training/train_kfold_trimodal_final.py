"""
train_kfold_trimodal_final.py
K折交叉验证 + 三模态融合 (RGB + TIR 前3通道 + MS)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pandas as pd

sys.path.append('.')
from datasets.dataset_drought import DroughtDataset

# ============================================================================
# 包装数据集 - 只取 TIR 的前 3 通道
# ============================================================================

class TrimodalDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        rgb, tir, ms, label = self.dataset[idx]
        
        # TIR 只取前 3 通道 (假设是 4 通道)
        if tir.shape[0] == 4:
            tir = tir[:3, :, :]
        
        return rgb, tir, ms, label

# ============================================================================
# 三模态融合模型
# ============================================================================

class TrimodalFusionNet(nn.Module):
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
        
        # TIR 编码器 (3通道)
        self.tir_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
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
        
        # 融合权重
        self.fusion_weights = nn.Sequential(
            nn.Linear(768, 128),
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
        rgb_feat = self.rgb_encoder(rgb).view(rgb.size(0), -1)
        tir_feat = self.tir_encoder(tir).view(tir.size(0), -1)
        ms_feat = self.ms_encoder(ms).view(ms.size(0), -1)
        
        concat = torch.cat([rgb_feat, tir_feat, ms_feat], dim=1)
        weights = self.fusion_weights(concat)
        
        fused = (weights[:, 0:1] * rgb_feat + 
                 weights[:, 1:2] * tir_feat + 
                 weights[:, 2:3] * ms_feat)
        
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
# 训练
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0
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
        
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    
    for rgb, tir, ms, labels in val_loader:
        rgb, tir, ms = rgb.to(device), tir.to(device), ms.to(device)
        
        logits, _ = model(rgb, tir, ms)
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        targets_all.extend(labels.numpy())
    
    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average='weighted', zero_division=0)
    
    return acc, f1, np.array(preds_all), np.array(targets_all)

# ============================================================================
# 主函数
# ============================================================================

def main():
    csv_path = "2025label_classic5.csv"
    data_root = "dataset/"
    epochs = 150
    batch_size = 4
    lr = 1e-4
    num_folds = 5
    
    os.makedirs("./models_trimodal_kfold", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print(f"🔗 三模态融合 K折交叉验证 (K={num_folds})")
    print("="*80)
    print("🌐 数据: RGB (3) + TIR (3/4) + MS (8)")
    print("="*80)
    
    # 加载数据
    print("\n📊 加载数据...")
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    
    print(f"✅ 数据集: {len(df)} 个样本")
    
    # K折
    print(f"\n🔄 K折交叉验证...\n")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_ensemble_preds = []
    all_ensemble_targets = np.array([], dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*80}")
        print(f"Fold {fold+1}/{num_folds} | Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*80}")
        
        # 生成子集 CSV
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_csv = f"/tmp/train_fold{fold}.csv"
        val_csv = f"/tmp/val_fold{fold}.csv"
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        
        # 创建数据集
        train_raw = DroughtDataset(train_csv, data_root=data_root, modalities=['rgb', 'tir', 'ms'], ids=list(train_df['id'].values))
        val_raw = DroughtDataset(val_csv, data_root=data_root, modalities=['rgb', 'tir', 'ms'], ids=list(val_df['id'].values))
        
        # 包装数据集 (只取 TIR 的前 3 通道)
        train_ds = TrimodalDatasetWrapper(train_raw)
        val_ds = TrimodalDatasetWrapper(val_raw)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 模型
        model = TrimodalFusionNet(num_classes=5)
        model.to(device)
        
        # 权重
        from collections import Counter
        train_labels = train_df['label'].values
        counts = Counter(train_labels)
        total = len(train_labels)
        weights = torch.tensor(
            [total / (5 * counts.get(i, 1)) for i in range(5)],
            dtype=torch.float, device=device
        )
        weights = weights / weights.sum() * 5
        
        # 损失
        focal_loss = FocalLoss(alpha=weights, gamma=2.0)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
        
        def criterion(outputs, targets):
            return 0.6 * focal_loss(outputs, targets) + 0.4 * ce_loss(outputs, targets)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 训练
        best_f1 = 0
        patience = 20
        patience_cnt = 0
        best_preds = None
        best_targets = None
        best_acc = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1, val_preds, val_targets = evaluate(model, val_loader, device)
            scheduler.step()
            
            if epoch % 30 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: train_acc={train_acc*100:.1f}%, val_acc={val_acc*100:.1f}%, val_f1={val_f1*100:.1f}%")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                patience_cnt = 0
                best_preds = val_preds
                best_targets = val_targets
                
                torch.save(model.state_dict(), f"./models_trimodal_kfold/fold{fold+1}_best.pth")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break
        
        fold_results.append({'fold': fold+1, 'acc': best_acc, 'f1': best_f1})
        all_ensemble_preds.extend(best_preds)
        all_ensemble_targets = np.concatenate([all_ensemble_targets, best_targets])
        
        print(f"✅ Fold {fold+1}: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%")
        
        # 清理
        os.remove(train_csv)
        os.remove(val_csv)
    
    # 结果
    print(f"\n{'='*80}")
    print("🎯 K折交叉验证总结")
    print(f"{'='*80}")
    
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    
    print(f"\n📊 单模型平均性能:")
    print(f"  准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    
    print(f"\n📋 各折详细结果:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']*100:.2f}%, F1={r['f1']*100:.2f}%")
    
    all_ensemble_preds = np.array(all_ensemble_preds)
    ensemble_acc = accuracy_score(all_ensemble_targets, all_ensemble_preds)
    ensemble_f1 = f1_score(all_ensemble_targets, all_ensemble_preds, average='weighted', zero_division=0)
    
    print(f"\n🔗 集成模型性能:")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    print(f"\n📈 相对第1阶段 (Acc=55.00%, F1=54.43%) 的改进:")
    print(f"  单模型准确率: {(np.mean(accs)-0.55)*100:+.2f}%")
    print(f"  单模型F1分数: {(np.mean(f1s)-0.5443)*100:+.2f}%")
    print(f"  集成准确率: {(ensemble_acc-0.55)*100:+.2f}%")
    print(f"  集成F1分数: {(ensemble_f1-0.5443)*100:+.2f}%")
    
    print(f"\n📋 集成模型分类详细报告:")
    print(classification_report(all_ensemble_targets, all_ensemble_preds,
                               target_names=[f'Level {i}' for i in range(5)],
                               digits=4))
    
    print(f"\n✅ 完成！所有模型已保存到 ./models_trimodal_kfold/")
    print("="*80)

if __name__ == '__main__':
    main()
