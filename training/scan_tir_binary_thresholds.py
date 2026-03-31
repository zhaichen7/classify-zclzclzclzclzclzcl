import os
import sys
import json
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders


# ============================================================================
# Model (must match train_tir_binary.py)
# ============================================================================

class TIRBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


# ============================================================================
# Metrics
# ============================================================================

def binary_metrics_from_preds(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (recall + specificity)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_acc": bal_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scan thresholds for TIR binary checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to binary_best.pth")
    parser.add_argument("--csv_path", required=True, help="path to mapped binary csv used in training")
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save_dir", required=True, help="directory to save threshold scan outputs")
    parser.add_argument("--thr_start", type=float, default=0.05)
    parser.add_argument("--thr_end", type=float, default=0.95)
    parser.add_argument("--thr_step", type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🔎 TIR 二分类 checkpoint 阈值扫描")
    print("=" * 80)
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"CSV Path   : {args.csv_path}")
    print(f"Data Root  : {args.data_root}")
    print(f"Save Dir   : {args.save_dir}")
    print(f"Thresholds : {args.thr_start} ~ {args.thr_end} step {args.thr_step}")
    print("=" * 80)

    print("\n📊 构建与训练时一致的验证集...")
    _, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['tir']
    )
    print("✅ 验证集加载完成")

    print("\n🧠 加载模型...")
    model = TIRBinaryClassifier(dropout=args.dropout).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("✅ 模型加载完成")

    print("\n📦 收集验证集概率...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for _, tir, _, labels in val_loader:
            tir = tir.to(device)
            logits = model(tir)
            probs = torch.softmax(logits, dim=1)[:, 1]   # positive class prob

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print(f"✅ Collected {len(all_labels)} validation samples")

    rows = []
    thr = args.thr_start
    while thr <= args.thr_end + 1e-12:
        preds = (all_probs >= thr).astype(int)
        metrics = binary_metrics_from_preds(preds, all_labels)
        rows.append({
            "threshold": round(thr, 4),
            **metrics
        })
        thr += args.thr_step

    df = pd.DataFrame(rows)

    # sort by F1 first, then balanced acc, then acc
    best_row = df.sort_values(
        by=["f1", "balanced_acc", "acc", "recall"],
        ascending=[False, False, False, False]
    ).iloc[0]

    top10 = df.sort_values(
        by=["f1", "balanced_acc", "acc", "recall"],
        ascending=[False, False, False, False]
    ).head(10)

    csv_out = os.path.join(args.save_dir, "threshold_scan.csv")
    json_out = os.path.join(args.save_dir, "best_threshold.json")
    top10_out = os.path.join(args.save_dir, "top10_thresholds.csv")

    df.to_csv(csv_out, index=False)
    top10.to_csv(top10_out, index=False)

    best_payload = {
        "checkpoint": args.checkpoint,
        "csv_path": args.csv_path,
        "data_root": args.data_root,
        "best_threshold": float(best_row["threshold"]),
        "best_metrics": {
            "acc": float(best_row["acc"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "specificity": float(best_row["specificity"]),
            "balanced_acc": float(best_row["balanced_acc"]),
            "tp": int(best_row["tp"]),
            "tn": int(best_row["tn"]),
            "fp": int(best_row["fp"]),
            "fn": int(best_row["fn"]),
        }
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✅ 阈值扫描完成")
    print("=" * 80)
    print(f"Best threshold: {best_row['threshold']:.4f}")
    print(
        f"Best metrics  : "
        f"Acc={best_row['acc']:.4f} | "
        f"F1={best_row['f1']:.4f} | "
        f"Precision={best_row['precision']:.4f} | "
        f"Recall={best_row['recall']:.4f} | "
        f"Specificity={best_row['specificity']:.4f} | "
        f"BalancedAcc={best_row['balanced_acc']:.4f}"
    )
    print(f"Saved full csv : {csv_out}")
    print(f"Saved top10 csv: {top10_out}")
    print(f"Saved best json: {json_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
