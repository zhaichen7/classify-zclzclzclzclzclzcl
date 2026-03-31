"""
热红外（TIR）单模态干旱分级训练脚本

说明：
- 复用仓库的 `datasets.dataset_drought.DroughtDataset` 数据加载与归一化逻辑
- 训练时只使用 TIR（3 通道），忽略 RGB/MS（但仍会读取，速度上比完全单模态慢一些）
"""

import os
import sys

# 让脚本能在项目根目录运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import DroughtClassifierRGBLite
from datasets.dataset_drought import build_dataloaders
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (_, tir, _, labels) in enumerate(pbar):
        tir = tir.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(tir)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {correct/total*100:.2f}%")
            sys.stdout.flush()

        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                         "acc": f"{correct/total*100:.2f}%"})

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validation", leave=False)
    for _, tir, _, labels in pbar:
        tir = tir.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(tir)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="TIR 单模态干旱分级训练")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="2025label_classic5.csv 路径")
    parser.add_argument("--data_root", type=str,
                        required=True, help="dataset 数据根目录")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--augment_train",
                        action="store_true", help="启用数据增强（翻转等）")
    parser.add_argument("--augmentation_factor", type=int,
                        default=0, help="增强倍数；>0 才会扩增数据集")
    parser.add_argument(
        "--no_balanced", action="store_true", help="关闭均衡分割（默认开启）")
    parser.add_argument("--normalize_method", type=str,
                        default="percentile", choices=["percentile", "minmax"])
    parser.add_argument("--dim", type=int, default=32, help="Lite 模型隐藏维度")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./models_tir_only")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "drought_best_tir_only.pth")

    print("=" * 60)
    print("TIR Single-Modal Drought Classification Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CSV:    {args.csv_path}")
    print(f"Data:   {args.data_root}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(
        f"Augment: {args.augment_train}, augmentation_factor: {args.augmentation_factor}")
    print(f"Balanced split: {not args.no_balanced}")
    print("=" * 60)

    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=args.augment_train,
        normalize_method=args.normalize_method,
        target_size=(args.target_size, args.target_size),
        balanced=(not args.no_balanced),
        augmentation_factor=args.augmentation_factor,
        modalities=['tir'],
    )

    model = DroughtClassifierRGBLite(
        dim=args.dim, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        sys.stdout.flush()
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        print(f"  Validating...")
        sys.stdout.flush()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"time={time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved best model: {best_path}")

    print(f"训练结束，最佳 val_acc={best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
