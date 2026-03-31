"""
train_trimodal_kfold_v2.py
K折交叉验证 + 三模态融合 (RGB + TIR + MS)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# 数据集
# ============================================================================

class TrimodalDataset(Dataset):
    def __init__(self, df, data_root='dataset/'):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        
        # 使用索引作为 ID
        sample_id = idx
        
        # 尝试多种文件名格式
        rgb_paths = [
            os.path.join(self.data_root, f"RGB_{sample_id}.npy"),
            os.path.join(self.data_root, f"rgb_{sample_id}.npy"),
        ]
        
        tir_paths = [
            os.path.join(self.data_root, f"TIR_{sample_id}.npy"),
            os.path.join(self.data_root, f"tir_{sample_id}.npy"),
        ]
        
        ms_paths = [
            os.path.join(self.data_root, f"MS_{sample_id}.npy"),
            os.path.join(self.data_root, f"ms_{sample_id}.npy"),
        ]
        
        # 找到存在的文件
        rgb_path = next((p for p in rgb_paths if os.path.exists(p)), None)
        tir_path = next((p for p in tir_paths if os.path.exists(p)), None)
        ms_path = next((p for p in ms_paths if os.path.exists(p)), None)
        
        if not all([rgb_path, tir_path, ms_path]):
            raise FileNotFoundError(f"缺少数据文件: RGB={rgb_path}, TIR={tir_path}, MS={ms_path}")
        
        # 加载数据
        rgb = np.load(rgb_path).astype(np.float32) / 255.0
        tir = np.load(tir_path).astype(np.float32) / 255.0
        ms = np.load(ms_path).astype(np.float32) / 255.0
        
        return torch.from_numpy(rgb), torch.from_numpy(tir), torch.from_numpy(ms), label

# ============================================================================
# 模型
# ============================================================================

class TrimodalFusionNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # RGB 编码器
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
        
        # TIR 编码器
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
        
        # MS 编码器
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb, tir, ms, labels = rgb.to(device), tir.to(device), ms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(rgb, tir, ms)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    targets_all = []
    
    pbar = tqdm(loader, leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb, tir, ms = rgb.to(device), tir.to(device), ms.to(device)
        
        logits, _ = model(rgb, tir, ms)
        preds_all.extend(logits.argmax(1).cpu().numpy())
        targets_all.extend(labels.numpy())
    
    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average='weighted', zero_division=0)
    
    return acc, f1

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
    
    # 加载数据
    print("\n📊 加载数据...")
    df = pd.read_csv(csv_path)
    dataset = TrimodalDataset(df, data_root=data_root)
    labels = df['label'].values
    
    print(f"✅ 数据集: {len(dataset)} 个样本")
    
    # K折
    print(f"\n🔄 K折交叉验证...\n")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = []
    all_targets = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold+1}/{num_folds} | Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 模型
        model = TrimodalFusionNet(num_classes=5)
        model.to(device)
        
        # 权重
        from collections import Counter
        train_labels = labels[train_idx]
        counts = Counter(train_labels)
        total = len(train_labels)
        weights = torch.tensor([total/(5*counts.get(i,1)) for i in range(5)], dtype=torch.float, device=device)
        weights = weights / weights.sum() * 5
        
        # 损失和优化器
        focal = FocalLoss(alpha=weights, gamma=2.0)
        ce = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)
        criterion = lambda o, t: 0.6*focal(o, t) + 0.4*ce(o, t)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 训练
        best_f1 = 0
        patience = 20
        patience_cnt = 0
        best_preds = None
        
        for epoch in range(1, epochs+1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            scheduler.step()
            
            if epoch % 30 == 0:
                print(f"  Epoch {epoch}: train={train_acc*100:.1f}%, val={val_acc*100:.1f}%, f1={val_f1*100:.1f}%")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                patience_cnt = 0
                
                torch.save(model.state_dict(), f"./models_trimodal_kfold/fold{fold+1}_best.pth")
                
                # 保存预测
                model.eval()
                with torch.no_grad():
                    preds = []
                    for rgb, tir, ms, _ in val_loader:
                        rgb, tir, ms = rgb.to(device), tir.to(device), ms.to(device)
                        logits, _ = model(rgb, tir, ms)
                        preds.extend(logits.argmax(1).cpu().numpy())
                    best_preds = np.array(preds)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break
        
        fold_results.append({'fold': fold+1, 'acc': best_acc, 'f1': best_f1})
        all_preds.extend(best_preds)
        all_targets.extend(labels[val_idx])
        
        print(f"✅ Fold {fold+1}: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%\n")
    
    # 结果
    print("\n" + "="*80)
    print("🎯 K折结果汇总")
    print("="*80)
    
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    
    print(f"\n单模型平均:")
    print(f"  准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    
    ensemble_acc = accuracy_score(all_targets, all_preds)
    ensemble_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print(f"\n集成模型:")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    print(f"\n相对第1阶段 (55%):")
    print(f"  单模型: {(np.mean(accs)-0.55)*100:+.2f}%")
    print(f"  集成: {(ensemble_acc-0.55)*100:+.2f}%")
    
    print("\n✅ 完成！")

if __name__ == '__main__':
    main()
