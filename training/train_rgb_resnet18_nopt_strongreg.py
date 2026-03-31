import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet18

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders


# ============================================================================
# Strong regularization helpers
# ============================================================================

def random_erasing_batch(x, p=0.25, scale=(0.02, 0.12)):
    """
    x: [B, C, H, W]
    """
    B, C, H, W = x.shape
    out = x.clone()
    for i in range(B):
        if torch.rand(1).item() < p:
            area = H * W
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            erase_h = max(1, int((erase_area) ** 0.5))
            erase_w = max(1, int((erase_area) ** 0.5))
            if erase_h >= H:
                erase_h = H - 1
            if erase_w >= W:
                erase_w = W - 1
            y = torch.randint(0, max(1, H - erase_h), (1,)).item()
            x0 = torch.randint(0, max(1, W - erase_w), (1,)).item()
            out[i, :, y:y+erase_h, x0:x0+erase_w] = 0.0
    return out


def add_gaussian_noise(x, std=0.03, p=0.30):
    if torch.rand(1).item() < p:
        noise = torch.randn_like(x) * std
        x = x + noise
        x = torch.clamp(x, 0.0, 1.0)
    return x


# ============================================================================
# Model
# ============================================================================

class RGBResNet18StrongReg(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        backbone = resnet18(weights=None)   # 关键：不用预训练

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),   # head 也缩小一点
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


# ============================================================================
# Train / Eval
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                    use_noise=True, noise_std=0.03, use_erasing=True, erasing_p=0.25):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for rgb, _, _, labels in pbar:
        rgb = rgb.to(device)
        labels = labels.to(device)

        if use_noise:
            rgb = add_gaussian_noise(rgb, std=noise_std, p=0.30)
        if use_erasing:
            rgb = random_erasing_batch(rgb, p=erasing_p)

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
    parser = argparse.ArgumentParser(description="RGB ResNet18 no-pretrain + stronger regularization")
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="./models_rgb_resnet18_nopt_strongreg")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--disable_noise", action="store_true")
    parser.add_argument("--disable_erasing", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 RGB ResNet18 去预训练 + 更强正则")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Noise std: {args.noise_std}")
    print(f"Use noise: {not args.disable_noise}")
    print(f"Use erasing: {not args.disable_erasing}")
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
    model = RGBResNet18StrongReg(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    print("\n🎯 创建损失函数...")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print("✅ Loss: CrossEntropyLoss")

    print("\n⚙️ 创建优化器...")
    optimizer = optim.AdamW(
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
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            use_noise=(not args.disable_noise),
            noise_std=args.noise_std,
            use_erasing=(not args.disable_erasing),
            erasing_p=0.25
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
