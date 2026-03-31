import os
import sys
import json
import random
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet18

from datasets.dataset_drought import build_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_binary_csv(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV 必须包含 label 列")
    uniq = sorted(df["label"].dropna().unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"二分类标签应当恰好有 2 个唯一值，当前是: {uniq}")
    label_map = {uniq[0]: 0, uniq[1]: 1}
    df = df.copy()
    df["label"] = df["label"].map(label_map).astype(int)

    mapped_csv = os.path.join(save_dir, "binary_mapped.csv")
    df.to_csv(mapped_csv, index=False)

    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

    return mapped_csv, label_map, df


def smooth_binary_targets(targets, smoothing=0.0):
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def binary_metrics_from_probs(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs >= threshold).astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = max(len(labels), 1)
    acc = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


class RGBBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        backbone = resnet18(weights=None)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, 1)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        rgb, tir, ms, labels = batch
        return rgb, tir, ms, labels
    raise ValueError("DataLoader 返回格式不是 (rgb, tir, ms, labels)")


def train_one_epoch(model, loader, criterion, optimizer, device, threshold=0.5, label_smoothing=0.0, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        rgb, _, _, labels = unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(rgb)
        targets = smooth_binary_targets(labels, label_smoothing)
        loss = criterion(logits, targets)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        total_loss += loss.item() * labels.size(0)

    metrics = binary_metrics_from_probs(all_probs, all_labels, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        rgb, _, _, labels = unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        logits = model(rgb)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())
        total_loss += loss.item() * labels.size(0)

    metrics = binary_metrics_from_probs(all_probs, all_labels, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RGB 单模态二分类")
    parser.add_argument("--csv_path", default="/home/zcl/addfuse1/2025label.csv")
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/dataset")
    parser.add_argument("--save_dir", default="/home/zcl/addfuse1/two/models_rgb_binary_run1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    mapped_csv, label_map, mapped_df = make_binary_csv(args.csv_path, args.save_dir)

    print("=" * 90)
    print("🚀 RGB 单模态二分类")
    print("=" * 90)
    print(f"Device: {device}")
    print(f"CSV: {mapped_csv}")
    print(f"Data root: {args.data_root}")
    print(f"Save dir: {args.save_dir}")
    print(f"Label map: {label_map}")
    print(f"Mapped label distribution: {dict(Counter(mapped_df['label'].tolist()))}")
    print("=" * 90)

    train_loader, val_loader = build_dataloaders(
        csv_path=mapped_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=args.seed,
        augment_train=True,
        balanced=True,
        modalities=['rgb']
    )

    model = RGBBinaryClassifier(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            threshold=args.threshold,
            label_smoothing=args.label_smoothing,
            grad_clip=1.0
        )
        val_metrics = validate(
            model, val_loader, criterion, device,
            threshold=args.threshold
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "tp": val_metrics["tp"],
            "tn": val_metrics["tn"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
        }
        history.append(row)

        print(
            f"✓ Epoch {epoch:03d}/{args.epochs} | "
            f"LR={row['lr']:.2e} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']*100:.2f}%"
        )
        print(
            f"  F1={row['val_f1']:.4f} | Precision={row['val_precision']:.4f} | "
            f"Recall={row['val_recall']:.4f} | Specificity={row['val_specificity']:.4f} | "
            f"TP={row['tp']} TN={row['tn']} FP={row['fp']} FN={row['fn']}"
        )

        improved = (row["val_f1"] > best_f1 + 1e-12) or (
            abs(row["val_f1"] - best_f1) <= 1e-12 and row["val_acc"] > best_acc
        )

        if improved:
            best_f1 = row["val_f1"]
            best_acc = row["val_acc"]
            best_epoch = epoch
            wait = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "precision": row["val_precision"],
                "recall": row["val_recall"],
                "specificity": row["val_specificity"],
                "threshold": args.threshold,
                "label_map": label_map,
                "modality": "rgb",
                "input_channels": 3,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "binary_best.pth"))
            print("  ✅ 保存最佳模型 (按 F1)")
        else:
            wait += 1
            print(f"  ⏳ 未提升，patience={wait}/{args.patience}")

        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "history.csv"), index=False)

        if wait >= args.patience:
            print(f"⚠️ 早停触发 (patience={args.patience})")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_f1": best_f1,
        "threshold": args.threshold,
        "label_map": label_map,
        "save_dir": args.save_dir,
    }
    with open(os.path.join(args.save_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("✅ 训练完成")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_acc*100:.2f}%")
    print(f"最佳 F1: {best_f1:.4f}")
    print(f"结果目录: {args.save_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
