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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import DroughtDataset


# ============================================================
# utils
# ============================================================

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
        raise ValueError(f"二分类标签必须恰好有 2 个唯一值，当前是: {uniq}")

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


def unpack_ms_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 4:
            rgb, tir, ms, labels = batch
            return ms, labels
        if len(batch) == 2:
            ms, labels = batch
            return ms, labels
    if isinstance(batch, dict):
        return batch["ms"], batch["label"]
    raise ValueError(f"无法识别 batch 格式: {type(batch)}")


def random_band_dropout(x, p=0.25, max_drop=2):
    if p <= 0 or max_drop <= 0:
        return x
    if random.random() > p:
        return x
    x = x.clone()
    b, c, _, _ = x.shape
    for i in range(b):
        k = random.randint(1, min(max_drop, c))
        drop_idx = random.sample(range(c), k)
        x[i, drop_idx] = 0.0
    return x


# ============================================================
# model
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class MSBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8,
            dim=48,
            num_blocks=[4, 6],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias"
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x).squeeze(1)
        return x


class MSBinarySEClassifier(nn.Module):
    def __init__(self, dropout=0.3, se_reduction=8):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8,
            dim=48,
            num_blocks=[4, 6],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias"
        )
        self.se = SEBlock(48, reduction=se_reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.se(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x).squeeze(1)
        return x


# ============================================================
# dataset helper
# ============================================================

def make_ms_dataset(csv_path, data_root, ids, train):
    """
    兼容你现在这套接口：
    DroughtDataset(csv_path, data_root, ids, augment=False, ...)
    如果局部参数名跟你仓库里有一点差别，这里已经尽量做了 fallback。
    """
    tried = []

    candidate_kwargs = [
        dict(
            csv_path=csv_path,
            data_root=data_root,
            ids=ids,
            augment=train,
            normalize_method="percentile",
            target_size=(224, 224),
            modalities=["ms"],
        ),
        dict(
            csv_path=csv_path,
            data_root=data_root,
            ids=ids,
            augment=train,
            modalities=["ms"],
        ),
        dict(
            csv_path=csv_path,
            data_root=data_root,
            ids=ids,
            modalities=["ms"],
        ),
    ]

    for kwargs in candidate_kwargs:
        try:
            return DroughtDataset(**kwargs)
        except TypeError as e:
            tried.append(str(e))
            continue

    raise RuntimeError(
        "DroughtDataset 构造失败，请检查 dataset_drought.py 里的参数名。\n"
        + "\n".join(tried)
    )


# ============================================================
# train / val
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device,
                    threshold=0.5, label_smoothing=0.0,
                    grad_clip=1.0, band_dropout_p=0.0, band_dropout_max=2):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        ms, labels = unpack_ms_batch(batch)
        ms = ms.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        ms = random_band_dropout(ms, p=band_dropout_p, max_drop=band_dropout_max)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ms)
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
        ms, labels = unpack_ms_batch(batch)
        ms = ms.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        logits = model(ms)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())
        total_loss += loss.item() * labels.size(0)

    metrics = binary_metrics_from_probs(all_probs, all_labels, threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MS 单模态二分类 5 折交叉验证")
    parser.add_argument("--csv_path", default="/home/zcl/addfuse1/2025label.csv")
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/dataset")
    parser.add_argument("--save_dir", default="/home/zcl/addfuse1/two/models_ms_binary_cv5_run1")

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--se_reduction", type=int, default=8)
    parser.add_argument("--band_dropout_p", type=float, default=0.25)
    parser.add_argument("--band_dropout_max", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    mapped_csv, label_map, mapped_df = make_binary_csv(args.csv_path, args.save_dir)
    ids = mapped_df["id"].tolist()
    labels = mapped_df["label"].tolist()

    print("=" * 100)
    print("🚀 MS Binary CV5")
    print("=" * 100)
    print(f"Device            : {device}")
    print(f"CSV               : {mapped_csv}")
    print(f"Data root         : {args.data_root}")
    print(f"Save dir          : {args.save_dir}")
    print(f"Use SE            : {args.use_se}")
    print(f"SE reduction      : {args.se_reduction}")
    print(f"Band dropout p    : {args.band_dropout_p if args.use_se else 0.0}")
    print(f"Band dropout max  : {args.band_dropout_max if args.use_se else 0}")
    print(f"Label map         : {label_map}")
    print(f"Label distribution: {dict(Counter(labels))}")
    print("=" * 100)

    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    fold_rows = []
    all_history_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, labels), start=1):
        print("\n" + "=" * 100)
        print(f"📂 Fold {fold_idx}/{args.num_folds}")
        print("=" * 100)

        train_ids = [ids[i] for i in train_idx]
        val_ids = [ids[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        print(f"Train size: {len(train_ids)} | label dist: {dict(Counter(train_labels))}")
        print(f"Val   size: {len(val_ids)} | label dist: {dict(Counter(val_labels))}")

        train_ds = make_ms_dataset(mapped_csv, args.data_root, train_ids, train=True)
        val_ds = make_ms_dataset(mapped_csv, args.data_root, val_ids, train=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        if args.use_se:
            model = MSBinarySEClassifier(
                dropout=args.dropout,
                se_reduction=args.se_reduction
            ).to(device)
        else:
            model = MSBinaryClassifier(
                dropout=args.dropout
            ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_f1 = -1.0
        best_acc = -1.0
        best_epoch = -1
        best_fold_metrics = None
        wait = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                threshold=args.threshold,
                label_smoothing=args.label_smoothing,
                grad_clip=1.0,
                band_dropout_p=(args.band_dropout_p if args.use_se else 0.0),
                band_dropout_max=args.band_dropout_max
            )
            val_metrics = validate(
                model, val_loader, criterion, device,
                threshold=args.threshold
            )
            scheduler.step()

            row = {
                "fold": fold_idx,
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
            all_history_rows.append(row)

            print(
                f"✓ Fold {fold_idx} Epoch {epoch:02d}/{args.epochs} | "
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
                best_fold_metrics = row.copy()
                wait = 0

                ckpt = {
                    "fold": fold_idx,
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
                    "modality": "ms",
                    "use_se": args.use_se,
                }
                torch.save(ckpt, os.path.join(args.save_dir, f"fold{fold_idx}_best.pth"))
                print("  ✅ 保存当前 fold 最佳模型 (按 F1)")
            else:
                wait += 1
                print(f"  ⏳ 未提升，patience={wait}/{args.patience}")

            if wait >= args.patience:
                print(f"⚠️ Fold {fold_idx} 早停触发")
                break

        fold_rows.append({
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_val_acc": best_fold_metrics["val_acc"],
            "best_f1": best_fold_metrics["val_f1"],
            "precision": best_fold_metrics["val_precision"],
            "recall": best_fold_metrics["val_recall"],
            "specificity": best_fold_metrics["val_specificity"],
            "tp": best_fold_metrics["tp"],
            "tn": best_fold_metrics["tn"],
            "fp": best_fold_metrics["fp"],
            "fn": best_fold_metrics["fn"],
        })

        pd.DataFrame(all_history_rows).to_csv(
            os.path.join(args.save_dir, "cv5_history.csv"),
            index=False
        )
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(args.save_dir, "cv5_fold_results.csv"),
            index=False
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "num_folds": args.num_folds,
        "use_se": args.use_se,
        "mean_acc": float(fold_df["best_val_acc"].mean()),
        "std_acc": float(fold_df["best_val_acc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "mean_f1": float(fold_df["best_f1"].mean()),
        "std_f1": float(fold_df["best_f1"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "mean_precision": float(fold_df["precision"].mean()),
        "mean_recall": float(fold_df["recall"].mean()),
        "mean_specificity": float(fold_df["specificity"].mean()),
    }

    with open(os.path.join(args.save_dir, "cv5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("✅ CV5 完成")
    print(f"Mean Acc: {summary['mean_acc']*100:.2f}% ± {summary['std_acc']*100:.2f}%")
    print(f"Mean F1 : {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    print(f"Mean Precision   : {summary['mean_precision']:.4f}")
    print(f"Mean Recall      : {summary['mean_recall']:.4f}")
    print(f"Mean Specificity : {summary['mean_specificity']:.4f}")
    print(f"结果目录: {args.save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
