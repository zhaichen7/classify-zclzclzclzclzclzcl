import os
import sys
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders


# ============================================================================
# Model
# ============================================================================

class RGBResNet18Classifier(nn.Module):
    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                print("✅ Loaded ImageNet pretrained ResNet18")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained weights: {e}")
                print("⚠️ Fallback to random init")
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


# ============================================================================
# Train / Eval
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for rgb, _, _, labels in pbar:
        rgb = rgb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total * 100:.2f}%"
        })

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for rgb, _, _, labels in pbar:
        rgb = rgb.to(device)
        labels = labels.to(device)

        logits = model(rgb)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total * 100:.2f}%"
        })

    class_accs = []
    for i in range(5):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)

    return total_loss / total, correct / total, class_accs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RGB single-modality ResNet18 baseline")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_rgb_resnet18_baseline")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 RGB单模态 + ResNet18 baseline")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Use pretrained: {not args.no_pretrained}")
    print(f"Use class weight: {args.use_class_weight}")
    print("=" * 80)

    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['rgb']
    )
    print("✅ 数据加载完成")

    print("\n🧠 创建模型...")
    model = RGBResNet18Classifier(
        dropout=args.dropout,
        use_pretrained=(not args.no_pretrained)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    class_weights = None
    if args.use_class_weight:
        print("\n⚖️ 计算类别权重...")
        train_labels = []
        for _, _, _, labels in train_loader:
            train_labels.extend(labels.numpy().tolist())
        label_counts = Counter(train_labels)
        total_samples = len(train_labels)

        weights = []
        for i in range(5):
            count = label_counts.get(i, 1)
            weights.append(total_samples / (5 * count))

        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        print(f"✅ class_weights: {class_weights.cpu().numpy()}")

    print("\n🎯 创建损失函数...")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    print("✅ Loss: CrossEntropyLoss")

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

    best_val_acc = 0.0
    best_epoch = 0
    best_class_accs = None
    patience_counter = 0

    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"\n✓ Epoch {epoch:3d}/{args.epochs} | LR={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )
        print("           Per-class acc: " + " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_class_accs = class_accs
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_accs": class_accs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, os.path.join(args.save_dir, "drought_best.pth"))

            print(f"           ✅ 保存最佳模型 (epoch {epoch})")
        else:
            patience_counter += 1
            print(f"           ⏳ 未提升，patience={patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n⚠️  早停触发 (patience={args.patience})")
                break

    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    if best_class_accs is not None:
        print(f"最佳类别准确率: {' | '.join([f'c{i}={best_class_accs[i]*100:.1f}%' for i in range(5)])}")
    print(f"模型保存路径: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
