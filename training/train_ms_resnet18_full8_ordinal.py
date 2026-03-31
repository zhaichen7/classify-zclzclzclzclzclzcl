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
# Utils
# ============================================================================

def labels_to_ordinal(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    ordinal = (labels.unsqueeze(1) > thresholds).float()
    return ordinal


def ordinal_logits_to_class(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).sum(dim=1).long()
    return preds


# ============================================================================
# Model
# ============================================================================

class ResNet18Ordinal8Ch(nn.Module):
    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet18(weights=None)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=8,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if use_pretrained:
                # 前3通道拷贝 ImageNet 权重
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # 后5通道用 RGB 平均权重初始化
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, 8):
                    new_conv.weight[:, c:c+1, :, :] = mean_w
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.conv1 = new_conv

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 4)   # 5类 -> 4个ordinal logits
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


# ============================================================================
# Train / Eval
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, pred_threshold):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        ordinal_targets = labels_to_ordinal(labels, num_classes=5)

        optimizer.zero_grad()
        logits = model(ms)
        loss = criterion(logits, ordinal_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = ordinal_logits_to_class(logits, threshold=pred_threshold)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total * 100:.2f}%"
        })

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch, pred_threshold):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        ordinal_targets = labels_to_ordinal(labels, num_classes=5)

        logits = model(ms)
        loss = criterion(logits, ordinal_targets)
        preds = ordinal_logits_to_class(logits, threshold=pred_threshold)

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
    parser = argparse.ArgumentParser(description="MS full8 + ResNet18-8ch + Ordinal")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_resnet18_full8_ordinal")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS full8 + ResNet18-8ch + Ordinal")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Pred threshold: {args.pred_threshold}")
    print(f"Use pos_weight: {args.use_pos_weight}")
    print(f"Use pretrained: {not args.no_pretrained}")
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
        modalities=['ms']
    )
    print("✅ 数据加载完成")

    print("\n🧠 创建模型...")
    model = ResNet18Ordinal8Ch(
        dropout=args.dropout,
        use_pretrained=(not args.no_pretrained)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    pos_weight = None
    if args.use_pos_weight:
        print("\n⚖️ 计算 ordinal pos_weight...")
        train_labels = []
        for _, _, _, labels in train_loader:
            train_labels.extend(labels.numpy().tolist())

        train_labels = torch.tensor(train_labels, dtype=torch.long)
        ordinal_targets = labels_to_ordinal(train_labels, num_classes=5)

        pos = ordinal_targets.sum(dim=0)
        neg = ordinal_targets.shape[0] - pos
        pos_weight = neg / (pos + 1e-6)
        pos_weight = pos_weight.to(device)
        print(f"✅ pos_weight: {pos_weight.cpu().numpy()}")

    print("\n🎯 创建损失函数...")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("✅ Loss: BCEWithLogitsLoss")

    print("\n⚙️ 创建优化器...")
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
            model, train_loader, criterion, optimizer, device, epoch, args.pred_threshold
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch, args.pred_threshold
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
                "pred_threshold": args.pred_threshold,
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
