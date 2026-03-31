"""
train_multiarch_comparison.py
多架构组合干旱分级对比训练脚本。

支持通过命令行参数指定各分支架构，方便不同架构组合的性能对比。

架构选项：
  --arch_rgb : restormer (默认) | efficientnet
  --arch_tir : restormer (默认) | densenet
  --arch_ms  : restormer (默认) | hybridsn

示例：
  # 基准：全 Restormer
  python training/train_multiarch_comparison.py \\
      --csv_path 2025label_classic5.csv \\
      --data_root dataset/ \\
      --arch_rgb restormer --arch_tir restormer --arch_ms restormer

  # 推荐组合：RGB(Restormer) + TIR(DenseNet) + MS(HybridSN)
  python training/train_multiarch_comparison.py \\
      --csv_path 2025label_classic5.csv \\
      --data_root dataset/ \\
      --arch_rgb restormer --arch_tir densenet --arch_ms hybridsn
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_multiarch import DroughtClassifierMultiArch
from datasets.dataset_drought import build_dataloaders


# ---------------------------------------------------------------------------
# 训练 / 验证辅助函数
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (rgb, tir, ms, labels) in enumerate(pbar):
        rgb    = rgb.to(device, non_blocking=True)
        tir    = tir.to(device, non_blocking=True)
        ms     = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        if (i + 1) % 50 == 0:
            print(
                f"  Batch {i+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {correct/total*100:.2f}%"
            )
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

    # 每类统计（用于打印每类精度）
    num_classes = None
    class_correct = None
    class_total = None

    pbar = tqdm(loader, desc="Validation", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb    = rgb.to(device, non_blocking=True)
        tir    = tir.to(device, non_blocking=True)
        ms     = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)

        if num_classes is None:
            num_classes = outputs.size(1)
            class_correct = torch.zeros(num_classes, device=device)
            class_total   = torch.zeros(num_classes, device=device)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += (preds[mask] == labels[mask]).sum()
            class_total[c]   += mask.sum()

    per_class_acc = {
        c: (class_correct[c] / class_total[c].clamp(min=1)).item()
        for c in range(num_classes or 5)
    }
    return (total_loss / max(total, 1),
            correct / max(total, 1),
            per_class_acc)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="多架构组合干旱分级对比训练")

    # 数据相关
    parser.add_argument("--csv_path", type=str, required=True,
                        help="2025label_classic5.csv 路径")
    parser.add_argument("--data_root", type=str, required=True,
                        help="dataset 数据根目录")
    parser.add_argument("--target_size", type=int, default=224,
                        help="图像缩放尺寸")
    parser.add_argument("--normalize_method", type=str,
                        default="percentile",
                        choices=["percentile", "minmax"])
    parser.add_argument("--no_balanced", action="store_true",
                        help="关闭均衡分割（默认开启）")
    parser.add_argument("--augment_train", action="store_true",
                        help="启用数据增强")
    parser.add_argument("--augmentation_factor", type=int, default=0,
                        help="增强倍数；>0 才会扩增数据集")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64,
                        help="各分支输出特征维度")

    # 架构选择
    parser.add_argument("--arch_rgb", type=str, default="restormer",
                        choices=["restormer", "efficientnet"],
                        help="RGB 分支架构")
    parser.add_argument("--arch_tir", type=str, default="restormer",
                        choices=["restormer", "densenet"],
                        help="TIR 分支架构")
    parser.add_argument("--arch_ms", type=str, default="restormer",
                        choices=["restormer", "hybridsn"],
                        help="MS 分支架构")

    # 输出
    parser.add_argument("--save_dir", type=str,
                        default="./models_multiarch",
                        help="模型保存目录")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 设备 & 输出目录
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    arch_tag = (f"rgb_{args.arch_rgb}_tir_{args.arch_tir}"
                f"_ms_{args.arch_ms}")
    best_path = os.path.join(args.save_dir,
                             f"drought_best_{arch_tag}.pth")

    print("=" * 70)
    print("多架构组合干旱分级对比训练")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"CSV        : {args.csv_path}")
    print(f"Data       : {args.data_root}")
    print(f"Epochs     : {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"arch_rgb   : {args.arch_rgb}")
    print(f"arch_tir   : {args.arch_tir}")
    print(f"arch_ms    : {args.arch_ms}")
    print(f"dim        : {args.dim}")
    print(f"Save to    : {best_path}")
    print("=" * 70)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 模型 / 优化器 / 调度器
    # ------------------------------------------------------------------
    model = DroughtClassifierMultiArch(
        arch_rgb=args.arch_rgb,
        arch_tir=args.arch_tir,
        arch_ms=args.arch_ms,
        dim=args.dim,
        num_classes=args.num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    best_val_acc = -1.0
    results = []  # 记录每 epoch 指标，方便后续对比

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        sys.stdout.flush()
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"time={elapsed:.1f}s"
        )
        per_class_str = " | ".join(
            f"c{c}={acc*100:.1f}%" for c, acc in per_class_acc.items())
        print(f"  Per-class val acc: {per_class_str}")
        sys.stdout.flush()

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class_acc": per_class_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "arch_rgb": args.arch_rgb,
                    "arch_tir": args.arch_tir,
                    "arch_ms":  args.arch_ms,
                    "dim": args.dim,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  已保存最佳模型 -> {best_path}  "
                  f"(val_acc={best_val_acc*100:.2f}%)")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # 训练结束汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("训练完成 — 性能汇总")
    print("=" * 70)
    print(f"架构组合  : RGB={args.arch_rgb} | TIR={args.arch_tir} | MS={args.arch_ms}")
    print(f"最佳 val_acc = {best_val_acc*100:.2f}%")
    print(f"最佳模型保存至: {best_path}")

    # 打印最后 5 epoch 趋势
    print("\n最后 5 个 epoch：")
    for r in results[-5:]:
        print(
            f"  Epoch {r['epoch']:03d} | "
            f"train_acc={r['train_acc']*100:.2f}% | "
            f"val_acc={r['val_acc']*100:.2f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
