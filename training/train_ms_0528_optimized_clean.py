"""
0528 专用 MS 单模态训练
基于 train_ms_optimized_v1.py 的核心思路：
- RestormerEncoder
- Focal Loss + Label Smoothing
- balanced split
- augment_train=True
但使用独立的 0528 MS loader，彻底绕开旧 get_file_paths
"""

import os
import sys
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought_0528_ms import (
    build_dataloaders_0528_ms,
    check_missing_0528_files,
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MSClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
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

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)

        outputs = model(ms)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    class_accs = []
    for i in range(5):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)

    return epoch_loss, epoch_acc, class_accs

def main():
    parser = argparse.ArgumentParser(description="0528 数据集 MS 单模态 optimized clean 训练")
    parser.add_argument("--csv_path", default="/home/zcl/addfuse1/2025label_classic5.csv")
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/0528")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--save_dir", default="./models_ms_0528_opt_clean")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--augmentation_factor", type=int, default=0)
    parser.add_argument("--val_per_class", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 0528 | MS单模态 | optimized clean")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"csv_path: {args.csv_path}")
    print(f"data_root: {args.data_root}")
    print(f"save_dir: {args.save_dir}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"gamma: {args.gamma}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"augmentation_factor: {args.augmentation_factor}")
    print(f"val_per_class: {args.val_per_class}")
    print("=" * 80)

    print("\n🔍 先检查 0528 文件路径...")
    missing = check_missing_0528_files(args.csv_path, args.data_root)
    print(f"missing count: {len(missing)}")
    if len(missing) > 0:
        for x in missing[:20]:
            print(x)
        raise RuntimeError("0528 数据仍有缺失路径，先别训练。")

    print("\n📊 构建 DataLoader...")
    train_loader, val_loader = build_dataloaders_0528_ms(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=42,
        augment_train=True,
        normalize_method="percentile",
        target_size=(224, 224),
        augmentation_factor=args.augmentation_factor,
        val_per_class=args.val_per_class,
    )
    print("✅ DataLoader 构建完成")

    print("\n🧠 创建模型...")
    encoder = RestormerEncoder(
        inp_channels=8,
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    model = MSClassifier(encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    print("\n⚖️ 计算类别权重...")
    train_labels = []
    for _, _, _, labels in train_loader:
        train_labels.extend(labels.numpy().tolist())

    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for i in range(5):
        count = label_counts.get(i, 1)
        weight = total_samples / (5 * count)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"✅ 类别权重: {class_weights.cpu().numpy()}")

    print("\n🎯 创建损失函数...")
    focal_loss = FocalLoss(alpha=class_weights, gamma=args.gamma)
    label_smooth_loss = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
        weight=class_weights
    )

    def criterion(outputs, targets):
        focal = focal_loss(outputs, targets)
        smooth = label_smooth_loss(outputs, targets)
        return 0.5 * focal + 0.5 * smooth

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)

    best_val_acc = 0.0
    best_epoch = 0
    best_class_accs = [0.0] * 5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(
            f"\n✓ Epoch {epoch:03d}/{args.epochs} | "
            f"LR={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )
        print("   Per-class acc: " + " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_class_accs = class_accs
            patience_counter = 0

            save_path = os.path.join(args.save_dir, "drought_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_accs": class_accs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, save_path)

            print(f"   ✅ 保存最佳模型: {save_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ 未提升，patience={patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n⚠️ 早停触发 (patience={args.patience})")
                break

    print("\n" + "=" * 80)
    print("✅ 训练完成")
    print("=" * 80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print("最佳类别准确率: " + " | ".join([f"c{i}={best_class_accs[i]*100:.1f}%" for i in range(5)]))
    print(f"模型保存路径: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
