"""
多光谱（MS）单模态干旱分级训练脚本

说明：
- 复用仓库的 `datasets.dataset_drought.DroughtDataset` 数据加载与归一化逻辑
- 训练时只使用 MS（8 通道），忽略 RGB/TIR（但仍会读取，速度上比完全单模态慢一些）
"""

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DroughtClassifierMSLite(nn.Module):
    """
    一个针对 MS(8通道) 的 Lite 单模态干旱分类器
    """

    def __init__(
        self,
        dim=32,
        num_blocks=(2, 3),
        heads=(1, 2),
        ffn_expansion_factor=2,
        num_classes=5,
        inp_channels=8,
    ):
        super().__init__()
        self.encoder_ms = RestormerEncoder(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=list(num_blocks),
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, ms):
        # ms: (B, 8, H, W)
        feat = self.encoder_ms(ms)  # (B, dim, H, W)
        pooled = self.global_pool(feat).view(feat.size(0), -1)  # (B, dim)
        return self.classifier(pooled)  # (B, num_classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(ms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

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
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(ms)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="MS 单模态干旱分级训练")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="2025label_classic5.csv 路径")
    parser.add_argument("--data_root", type=str,
                        required=True, help="dataset 数据根目录")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
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
    parser.add_argument("--save_dir", type=str, default="./models_ms_only")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "drought_best_ms_only.pth")

    print("=" * 60)
    print("MS 单模态干旱分级训练")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CSV:    {args.csv_path}")
    print(f"Data:   {args.data_root}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(
        f"Augment: {args.augment_train}, augmentation_factor: {args.augmentation_factor}")
    print(f"Balanced split: {args.balanced}")
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
    )

    model = DroughtClassifierMSLite(
        dim=args.dim, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
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
            print(f"✅ 已保存最佳模型: {best_path}")

    print(f"训练结束，最佳 val_acc={best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
