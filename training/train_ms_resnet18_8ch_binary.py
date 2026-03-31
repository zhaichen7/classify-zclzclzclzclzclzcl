import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet18

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_drought import build_dataloaders


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    # 0 -> s/2, 1 -> 1-s/2
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def make_binary_csv(csv_path: str, save_dir: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"CSV 缺少 label 列: {csv_path}")

    uniq = sorted(df["label"].dropna().unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"binary 脚本要求 CSV 里恰好 2 个类别，当前是: {uniq}")

    if set(uniq) == {0, 1}:
        mapping = {0: 0, 1: 1}
        out_df = df.copy()
    else:
        mapping = {uniq[0]: 0, uniq[1]: 1}
        out_df = df.copy()
        out_df["label"] = out_df["label"].map(mapping).astype(int)

    out_csv = os.path.join(save_dir, "binary_mapped.csv")
    out_df.to_csv(out_csv, index=False)

    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "original_labels": uniq,
                "mapping": {str(k): int(v) for k, v in mapping.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_csv, mapping, out_df


def unpack_ms_batch(batch):
    """
    兼容几种常见返回形式：
    1) (rgb, tir, ms, labels)
    2) (ms, labels)
    3) {"ms": ..., "label": ...} / {"ms": ..., "labels": ...}
    """
    if isinstance(batch, dict):
        ms = batch.get("ms", None)
        labels = batch.get("label", batch.get("labels", None))
        if ms is None or labels is None:
            raise ValueError("dict batch 里没找到 ms / label(s)")
        return ms, labels

    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            _, _, ms, labels = batch
            return ms, labels
        if len(batch) == 2:
            ms, labels = batch
            return ms, labels

    raise ValueError(f"不支持的 batch 格式: {type(batch)}")


def prepare_ms_tensor(ms: torch.Tensor) -> torch.Tensor:
    ms = ms.float()
    # 兼容 [B, H, W, C] -> [B, C, H, W]
    if ms.ndim == 4 and ms.shape[-1] == 8 and ms.shape[1] != 8:
        ms = ms.permute(0, 3, 1, 2).contiguous()
    return ms


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_true = y_true.astype(np.int64).reshape(-1)
    y_prob = y_prob.astype(np.float32).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

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


def try_build_dataloaders(csv_path, data_root, batch_size, num_workers, seed):
    """
    尽量兼容你现有 repo 里的 build_dataloaders 形参。
    """
    candidates = [
        dict(
            csv_path=csv_path,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_train=True,
            balanced=True,
            augmentation_factor=0,
            modalities=["ms"],
            test_size=0.2,
            random_state=seed,
        ),
        dict(
            csv_path=csv_path,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_train=True,
            balanced=True,
            augmentation_factor=0,
            modalities=["ms"],
            val_ratio=0.2,
            seed=seed,
        ),
        dict(
            csv_path=csv_path,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_train=True,
            balanced=True,
            augmentation_factor=0,
            modalities=["ms"],
        ),
    ]

    last_err = None
    for kwargs in candidates:
        try:
            out = build_dataloaders(**kwargs)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return out[0], out[1]
            raise ValueError("build_dataloaders 返回值异常，无法解析 train/val loader")
        except TypeError as e:
            last_err = e
            continue

    raise RuntimeError(f"build_dataloaders 调用失败，请按你 repo 里的真实签名微调。最后错误: {last_err}")


# =========================================================
# Model
# =========================================================
class ResNet18MSBinary(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            8, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, threshold, label_smoothing):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        ms, labels = unpack_ms_batch(batch)
        ms = prepare_ms_tensor(ms).to(device, non_blocking=True)
        labels = labels.view(-1).float().to(device, non_blocking=True)

        logits = model(ms)
        smooth_targets = smooth_binary_targets(labels, label_smoothing)
        loss = criterion(logits, smooth_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        running_loss += loss.item() * ms.size(0)
        all_probs.append(probs)
        all_targets.append(targets)

        cur_metrics = binary_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_probs),
            threshold=threshold,
        )
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{cur_metrics['acc']*100:.2f}%")

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    metrics = binary_metrics(all_targets, all_probs, threshold=threshold)
    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, threshold):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        ms, labels = unpack_ms_batch(batch)
        ms = prepare_ms_tensor(ms).to(device, non_blocking=True)
        labels = labels.view(-1).float().to(device, non_blocking=True)

        logits = model(ms)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.cpu().numpy()

        running_loss += loss.item() * ms.size(0)
        all_probs.append(probs)
        all_targets.append(targets)

        cur_metrics = binary_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_probs),
            threshold=threshold,
        )
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{cur_metrics['acc']*100:.2f}%")

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    metrics = binary_metrics(all_targets, all_probs, threshold=threshold)
    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return epoch_loss, metrics


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("MS ResNet18-8ch Binary Baseline")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ensure_dir(args.save_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print("MS ResNet18-8ch Binary Baseline")
    print(f"device: {device}")
    print(f"save_dir: {args.save_dir}")
    print("=" * 90)

    binary_csv, mapping, mapped_df = make_binary_csv(args.csv_path, args.save_dir)
    print(f"binary csv: {binary_csv}")
    print(f"label mapping: {mapping}")
    print("mapped label counts:")
    print(mapped_df["label"].value_counts().sort_index().to_dict())

    train_loader, val_loader = try_build_dataloaders(
        csv_path=binary_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = ResNet18MSBinary(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=max(args.lr * 0.1, 1e-6),
    )

    history = []
    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=args.threshold,
            label_smoothing=args.label_smoothing,
        )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "val_loss": val_loss,
            "val_acc": val_metrics["acc"],
            "f1": val_metrics["f1"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "specificity": val_metrics["specificity"],
            "tp": val_metrics["tp"],
            "tn": val_metrics["tn"],
            "fp": val_metrics["fp"],
            "fn": val_metrics["fn"],
            "seconds": time.time() - start,
        }
        history.append(row)

        print(
            f"✓ Epoch {epoch:03d}/{args.epochs} | LR={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['acc']*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['acc']*100:.2f}%"
        )
        print(
            f"  F1={val_metrics['f1']:.4f} | Precision={val_metrics['precision']:.4f} | "
            f"Recall={val_metrics['recall']:.4f} | Specificity={val_metrics['specificity']:.4f} | "
            f"TP={val_metrics['tp']} TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']}"
        )

        improved = (
            (val_metrics["f1"] > best_f1 + 1e-8) or
            (abs(val_metrics["f1"] - best_f1) <= 1e-8 and val_metrics["acc"] > best_acc + 1e-8)
        )

        if improved:
            best_f1 = val_metrics["f1"]
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = os.path.join(args.save_dir, "binary_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "label_mapping": mapping,
                    "best_metrics": val_metrics,
                },
                ckpt_path,
            )

            with open(os.path.join(args.save_dir, "best_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_val_acc": best_acc,
                        "best_f1": best_f1,
                        "best_metrics": val_metrics,
                        "label_mapping": {str(k): int(v) for k, v in mapping.items()},
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"  ✅ 保存最佳模型: {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  ⏳ 未提升，patience={patience_counter}/{args.patience}")

        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "history.csv"), index=False)

        if patience_counter >= args.patience:
            print(f"⚠️ 早停触发 (patience={args.patience})")
            break

    print("=" * 90)
    print("✅ 训练完成")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_acc*100:.2f}%")
    print(f"最佳 F1: {best_f1:.4f}")
    print(f"结果目录: {args.save_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
