import os
import sys
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders


# ============================================================================
# Utils
# ============================================================================

def labels_to_ordinal(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    5类 -> 4个阈值:
    y=0 -> [0,0,0,0]
    y=1 -> [1,0,0,0]
    y=2 -> [1,1,0,0]
    y=3 -> [1,1,1,0]
    y=4 -> [1,1,1,1]
    """
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)  # [1,4]
    ordinal = (labels.unsqueeze(1) > thresholds).float()
    return ordinal


def ordinal_logits_to_class(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).sum(dim=1).long()
    return preds


# ============================================================================
# Model
# ============================================================================

class MSOrdinalClassifier(nn.Module):
    def __init__(self, encoder, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4)   # 5类 -> 4个ordinal logits
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
    parser = argparse.ArgumentParser(description="MS full8 ordinal classification training")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_full8_ordinal")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--use_pos_weight", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS单模态 full8 + ordinal classification")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Pred threshold: {args.pred_threshold}")
    print(f"Use pos_weight: {args.use_pos_weight}")
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
    encoder = RestormerEncoder(
        inp_channels=8,
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    model = MSOrdinalClassifier(encoder, dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    pos_weight = None
    if args.use_pos_weight:
        print("\n⚖️ 计算 ordinal pos_weight...")
        train_labels = []
        for _, _, _, labels in train_loader:
            train_labels.extend(labels.numpy().tolist())

        train_labels = torch.tensor(train_labels, dtype=torch.long)
        ordinal_targets = labels_to_ordinal(train_labels, num_classes=5)  # [N,4]

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

    print(f"✅ Optimizer: Adam")
    print(f"✅ Scheduler: CosineAnnealingLR")

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
        class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
        print(f"           Per-class acc: {class_str}")

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
                "pred_threshold": args.pred_threshold,
            }, save_path)

            print(f"           ✅ 保存最佳模型 (epoch {epoch})")
        else:
            patience_counter += 1
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
