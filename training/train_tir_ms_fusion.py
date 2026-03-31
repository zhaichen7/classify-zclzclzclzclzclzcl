"""
train_tir_ms_fusion.py
TIR + MS 融合 - 决策级融合 + 多任务学习
TIR对干旱检测更敏感，预期效果更好
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
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_drought import build_dataloaders

class TIREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

class MSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

class TIRMSFusion(nn.Module):
    """TIR+MS双头融合"""
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.tir_encoder = TIREncoder()
        self.ms_encoder = MSEncoder()
        
        # TIR独立分类头
        self.tir_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # MS独立分类头
        self.ms_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 融合权重 (可学习)
        self.tir_weight = nn.Parameter(torch.tensor(1.0))
        self.ms_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, tir, ms):
        # 编码
        tir_feat = self.tir_encoder(tir)
        ms_feat = self.ms_encoder(ms)
        
        # 独立分类
        tir_logits = self.tir_classifier(tir_feat)
        ms_logits = self.ms_classifier(ms_feat)
        
        # 加权融合 (TIR权重更大，因为对干旱更敏感)
        tir_w = torch.sigmoid(self.tir_weight)
        ms_w = torch.sigmoid(self.ms_weight)
        total_w = tir_w + ms_w
        tir_w = tir_w / total_w
        ms_w = ms_w / total_w
        
        fused_logits = tir_w * tir_logits + ms_w * ms_logits
        
        return fused_logits, tir_logits, ms_logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for _, tir, ms, labels in pbar:
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        fused_logits, tir_logits, ms_logits = model(tir, ms)
        
        # 多任务损失
        loss_fused = criterion(fused_logits, labels)
        loss_tir = criterion(tir_logits, labels) * 0.4  # TIR权重更大
        loss_ms = criterion(ms_logits, labels) * 0.2
        loss = loss_fused + loss_tir + loss_ms
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = fused_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False)
    for _, tir, ms, labels in pbar:
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        fused_logits, tir_logits, ms_logits = model(tir, ms)
        loss_fused = criterion(fused_logits, labels)
        loss_tir = criterion(tir_logits, labels) * 0.4
        loss_ms = criterion(ms_logits, labels) * 0.2
        loss = loss_fused + loss_tir + loss_ms
        
        total_loss += loss.item() * labels.size(0)
        preds = fused_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()
    
    class_accs = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(5)]
    return total_loss / total, correct / total, class_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_tir_ms_fusion")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🚀 TIR + MS 融合 (决策级融合 + 多任务学习)")
    print("="*80)
    
    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=False,
        balanced=False,
        modalities=['tir', 'ms']
    )
    print(f"✅ 数据加载完成")
    
    print("\n🧠 创建模型...")
    model = TIRMSFusion(num_classes=5)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    print("\n⚖️  计算类别权重...")
    labels_list = []
    for _, _, _, labels in train_loader:
        labels_list.extend(labels.numpy())
    
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*80)
    print("🚀 开始训练...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        
        if epoch % 12 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: LR={lr:.2e}, train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
            class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
            print(f"         Per-class: {class_str}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
            }, os.path.join(args.save_dir, 'fusion_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ 融合训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对MS单模态 (55.29%) 提升: {(best_val_acc - 0.5529)*100:+.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'fusion_best.pth')}")
    print("="*80)

if __name__ == '__main__':
    main()
