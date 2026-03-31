"""
train_trimodal_kfold.py
K折交叉验证 + 三模态融合 (RGB + TIR + MS)
预期: 55% → 70-80%

核心思路:
  1. K折充分利用所有数据
  2. 三个模态各有编码器
  3. 多头注意力融合
  4. 每一折都是独立的模型，最后集成投票
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders, DroughtDataset

# ============================================================================
# 三模态融合网络
# ============================================================================

class TrimodalFusionNet(nn.Module):
    """
    三模态融合：RGB + TIR + MS
    
    RGB (3)  ──┐
    TIR (1)  ──┼──> 各自编码器 ──> 特征融合 (多头注意力) ──> 分类
    MS (8)   ──┘
    """
    def __init__(self, num_classes=5):
        super().__init__()
        
        # ========== RGB 编码器 ==========
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== TIR 编码器 ==========
        self.tir_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== MS 编码器 ==========
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # ========== 多头注意力融合 ==========
        # 每个模态 256 维，3 个模态，总共 768 维
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        
        # 三个模态的融合权重学习
        self.fusion_weights = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )
        
        # ========== 分类头 ==========
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, tir, ms):
        """
        rgb: (B, 3, 224, 224)
        tir: (B, 1, 224, 224)
        ms:  (B, 8, 224, 224)
        """
        # 编码
        rgb_feat = self.rgb_encoder(rgb).view(rgb.size(0), -1)  # (B, 256)
        tir_feat = self.tir_encoder(tir).view(tir.size(0), -1)  # (B, 256)
        ms_feat = self.ms_encoder(ms).view(ms.size(0), -1)      # (B, 256)
        
        # 堆叠特征用于多头注意力
        stacked_feats = torch.stack([rgb_feat, tir_feat, ms_feat], dim=1)  # (B, 3, 256)
        
        # 多头注意力融合
        attn_output, _ = self.multihead_attn(stacked_feats, stacked_feats, stacked_feats)  # (B, 3, 256)
        
        # 学习融合权重
        concat_feats = torch.cat([rgb_feat, tir_feat, ms_feat], dim=1)  # (B, 768)
        fusion_weights = self.fusion_weights(concat_feats)  # (B, 3)
        
        # 加权融合
        fusion_weights = fusion_weights.unsqueeze(2)  # (B, 3, 1)
        fused_feat = (attn_output * fusion_weights).sum(dim=1)  # (B, 256)
        
        # 分类
        logits = self.classifier(fused_feat)  # (B, 5)
        
        return logits, fusion_weights.squeeze(2)

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
# 训练和评估函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, leave=False, desc="Train")
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(val_loader, leave=False, desc="Val")
    for rgb, tir, ms, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        logits, _ = model(rgb, tir, ms)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    acc = correct / total
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return total_loss / total, acc, f1

# ============================================================================
# 主函数 - K折交叉验证
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="三模态融合 K折交叉验证")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_trimodal_kfold")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print(f"🔗 三模态融合 K折交叉验证 (K={args.num_folds})")
    print("="*70)
    print(f"Device: {device}")
    print(f"融合: RGB + TIR + MS")
    print("="*70)
    
    # 加载数据集
    print("\n📊 加载数据集...")
    df = pd.read_csv(args.csv_path)
    dataset = DroughtDataset(
        df,
        data_root=args.data_root,
        modalities=['rgb', 'tir', 'ms'],
        ids=None
    )
    labels = df['label'].values
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    # K折划分
    print(f"\n🔄 进行 {args.num_folds} 折交叉验证...\n")
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_fold_preds = []  # 用于最后的集成投票
    all_fold_targets = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{args.num_folds}")
        print(f"{'='*70}")
        
        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )
        
        print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
        
        # 创建模型
        model = TrimodalFusionNet(num_classes=5)
        model.to(device)
        
        # 计算类别权重
        train_labels = labels[train_idx]
        from collections import Counter
        label_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = []
        for i in range(5):
            count = label_counts.get(i, 1)
            weight = total / (5 * count)
            class_weights.append(weight)
        
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # 损失函数
        focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        label_smooth = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
        
        def criterion(outputs, targets):
            return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth(outputs, targets)
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 训练循环
        best_val_acc = 0
        best_f1 = 0
        patience_counter = 0
        best_preds = None
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            if epoch % 20 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: train_acc={train_acc*100:.1f}%, val_acc={val_acc*100:.1f}%, val_f1={val_f1*100:.1f}%")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存最佳模型和预测
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(args.save_dir, f'fold{fold+1}_best.pth'))
                
                # 保存最佳预测用于集成
                model.eval()
                with torch.no_grad():
                    fold_preds = []
                    for rgb, tir, ms, _ in val_loader:
                        rgb = rgb.to(device)
                        tir = tir.to(device)
                        ms = ms.to(device)
                        logits, _ = model(rgb, tir, ms)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        fold_preds.extend(preds)
                    best_preds = np.array(fold_preds)
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"早停 (Fold {fold+1}, Epoch {epoch})")
                    break
        
        fold_results.append({
            'fold': fold + 1,
            'acc': best_val_acc,
            'f1': best_f1
        })
        
        all_fold_preds.append(best_preds)
        if all_fold_targets is None:
            all_fold_targets = labels[val_idx]
        
        print(f"✅ Fold {fold+1} 完成: Acc={best_val_acc*100:.2f}%, F1={best_f1*100:.2f}%")
    
    # 集成投票
    print(f"\n{'='*70}")
    print("🔗 K折集成投票")
    print(f"{'='*70}")
    
    # 对齐预测和标签
    ensemble_preds = np.concatenate(all_fold_preds)
    ensemble_targets = all_fold_targets
    
    ensemble_acc = accuracy_score(ensemble_targets, ensemble_preds)
    ensemble_f1 = f1_score(ensemble_targets, ensemble_preds, average='weighted', zero_division=0)
    
    print(f"\n{args.num_folds}折集成结果:")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    # 总结
    print(f"\n{'='*70}")
    print("🎯 K折交叉验证总结")
    print(f"{'='*70}")
    
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    
    print(f"\n单模型性能:")
    print(f"  准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    
    print(f"\n各折详细结果:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']*100:.2f}%, F1={r['f1']*100:.2f}%")
    
    print(f"\n集成模型性能:")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    print(f"\n与第1阶段对比 (55%):")
    print(f"  单模型平均: {(np.mean(accs)-0.55)*100:+.2f}%")
    print(f"  集成模型:  {(ensemble_acc-0.55)*100:+.2f}%")
    
    print(f"\n✅ 所有模型已保存到 {args.save_dir}/")
    print("="*70)
    
    # 详细分类报告
    print(f"\n分类详细报告:")
    print(classification_report(ensemble_targets, ensemble_preds,
                               target_names=[f'Level {i}' for i in range(5)]))

if __name__ == '__main__':
    main()
