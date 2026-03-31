"""
train_ms_kfold_pretrained.py
第5阶段：K折交叉验证 + ResNet50预训练模型
预期：55% → 75-85%
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import DroughtDataset

# ============================================================================
# 1. 使用预训练 ResNet50
# ============================================================================

class ResNet50Classifier(nn.Module):
    """基于预训练 ResNet50 的分类器"""
    def __init__(self, num_classes=5, in_channels=8):
        super().__init__()
        
        # 加载预训练 ResNet50
        import torchvision.models as models
        resnet50 = models.resnet50(pretrained=True)
        
        # 修改输入层以支持8通道（多光谱）
        original_conv = resnet50.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 复制预训练权重到新的卷积层（平均8个通道）
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        
        # 冻结前面的层，只微调最后几层
        resnet50.conv1 = self.conv1
        
        # 冻结 layer1、layer2
        for param in resnet50.layer1.parameters():
            param.requires_grad = False
        for param in resnet50.layer2.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# 2. Focal Loss
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
# 3. 训练函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, leave=False)
    for _, _, ms, labels in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(val_loader, leave=False)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return total_loss / total, correct / total, f1

# ============================================================================
# 4. K折交叉验证
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_kfold")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print(f"🚀 K折交叉验证 (K={args.num_folds}) + ResNet50预训练")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs per fold: {args.epochs}")
    print("="*70)
    
    # 加载数据集
    print("\n📊 加载数据集...")
    df = pd.read_csv(args.csv_path)
    dataset = DroughtDataset(df, data_root=args.data_root, modalities=['ms'])
    labels = df['label'].values
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    # K折划分
    print(f"\n🔄 进行 {args.num_folds} 折交叉验证...")
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{args.num_folds}")
        print(f"{'='*70}")
        
        # 创建数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
        
        # 创建模型
        model = ResNet50Classifier(num_classes=5, in_channels=8)
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
        
        # 优化器（只微调后面的层）
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 训练
        best_val_acc = 0
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}: train_acc={train_acc*100:.1f}%, val_acc={val_acc*100:.1f}%, val_f1={val_f1*100:.1f}%")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(args.save_dir, f'fold{fold+1}_best.pth'))
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"早停 (Fold {fold+1})")
                    break
        
        fold_results.append({
            'fold': fold + 1,
            'acc': best_val_acc,
            'f1': best_f1
        })
        
        print(f"Fold {fold+1} 最佳准确率: {best_val_acc*100:.2f}%, F1: {best_f1*100:.2f}%")
    
    # 总结
    print(f"\n{'='*70}")
    print("🎯 K折交叉验证总结")
    print(f"{'='*70}")
    
    accs = [r['acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    
    print(f"\n准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    
    print(f"\n各折详细结果:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']*100:.2f}%, F1={r['f1']*100:.2f}%")
    
    print(f"\n✅ 模型已保存到 {args.save_dir}/")

if __name__ == '__main__':
    main()
