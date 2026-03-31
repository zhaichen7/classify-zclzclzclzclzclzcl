import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders


# ============================================================================
# Utils
# ============================================================================

def infer_label_col(df):
    for cand in ["label", "labels", "class", "target", "y", "binary_label"]:
        if cand in df.columns:
            return cand
    non_id_cols = [c for c in df.columns if c.lower() != "id"]
    if len(non_id_cols) == 1:
        return non_id_cols[0]
    raise ValueError(f"无法自动识别标签列，当前列名: {df.columns.tolist()}")

def prepare_binary_csv(src_csv, dst_csv):
    df = pd.read_csv(src_csv)
    label_col = infer_label_col(df)

    uniq = sorted(df[label_col].dropna().unique().tolist())
    if len(uniq) != 2:
        raise ValueError(f"二分类标签数量不是2个，而是 {len(uniq)} 个: {uniq}")

    mapping = {uniq[0]: 0, uniq[1]: 1}
    df[label_col] = df[label_col].map(mapping).astype(int)

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df.to_csv(dst_csv, index=False)
    return mapping, len(df)

def binary_metrics_from_preds(preds, labels):
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)

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
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================================
# Models
# ============================================================================

class MSBinaryClassifier(nn.Module):
    def __init__(self, encoder, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MS + TIR late fusion for binary classification")
    parser.add_argument("--ms_ckpt", required=True)
    parser.add_argument("--tir_ckpt", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/dataset")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_start", type=float, default=0.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_step", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ms_dropout", type=float, default=0.3)
    parser.add_argument("--tir_dropout", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    mapped_csv = os.path.join(args.save_dir, "binary_mapped.csv")
    mapping, n_rows = prepare_binary_csv(args.csv_path, mapped_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS + TIR late fusion")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"MS ckpt     : {args.ms_ckpt}")
    print(f"TIR ckpt    : {args.tir_ckpt}")
    print(f"Original csv: {args.csv_path}")
    print(f"Mapped csv  : {mapped_csv}")
    print(f"Label map   : {mapping}")
    print(f"Rows        : {n_rows}")
    print(f"Save dir    : {args.save_dir}")
    print(f"Weight scan : {args.weight_start} ~ {args.weight_end} step {args.weight_step}")
    print("=" * 80)

    print("\n📊 加载验证集...")
    _, val_loader = build_dataloaders(
        csv_path=mapped_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['tir', 'ms']
    )
    print("✅ 验证集加载完成")

    print("\n🧠 加载 MS 模型...")
    ms_encoder = RestormerEncoder(
        inp_channels=8,
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    ms_model = MSBinaryClassifier(ms_encoder, dropout=args.ms_dropout).to(device)
    ms_ckpt = torch.load(args.ms_ckpt, map_location=device)
    ms_model.load_state_dict(ms_ckpt["model_state_dict"])
    ms_model.eval()
    print("✅ MS 模型加载完成")

    print("\n🧠 加载 TIR 模型...")
    tir_model = TIRBinaryClassifier(dropout=args.tir_dropout).to(device)
    tir_ckpt = torch.load(args.tir_ckpt, map_location=device)
    tir_model.load_state_dict(tir_ckpt["model_state_dict"])
    tir_model.eval()
    print("✅ TIR 模型加载完成")

    print("\n📦 收集验证集概率...")
    ms_probs_all = []
    tir_probs_all = []
    labels_all = []

    with torch.no_grad():
        for _, tir, ms, labels in val_loader:
            tir = tir.to(device)
            ms = ms.to(device)

            ms_logits = ms_model(ms)
            tir_logits = tir_model(tir)

            ms_probs = torch.softmax(ms_logits, dim=1)[:, 1]
            tir_probs = torch.softmax(tir_logits, dim=1)[:, 1]

            ms_probs_all.append(ms_probs.cpu())
            tir_probs_all.append(tir_probs.cpu())
            labels_all.append(labels.cpu())

    ms_probs_all = torch.cat(ms_probs_all).numpy()
    tir_probs_all = torch.cat(tir_probs_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    print(f"✅ Collected {len(labels_all)} validation samples")

    rows = []
    weight = args.weight_start
    while weight <= args.weight_end + 1e-12:
        ms_w = round(weight, 4)
        tir_w = round(1.0 - weight, 4)

        fused_probs = ms_w * ms_probs_all + tir_w * tir_probs_all
        preds = (fused_probs >= args.threshold).astype(int)

        metrics = binary_metrics_from_preds(preds, labels_all)

        rows.append({
            "ms_weight": ms_w,
            "tir_weight": tir_w,
            "threshold": args.threshold,
            **metrics
        })

        weight += args.weight_step

    df = pd.DataFrame(rows)

    best_row = df.sort_values(
        by=["f1", "acc", "balanced_acc", "recall"],
        ascending=[False, False, False, False]
    ).iloc[0]

    top10 = df.sort_values(
        by=["f1", "acc", "balanced_acc", "recall"],
        ascending=[False, False, False, False]
    ).head(10)

    csv_out = os.path.join(args.save_dir, "fusion_scan.csv")
    top10_out = os.path.join(args.save_dir, "top10_weights.csv")
    best_json_out = os.path.join(args.save_dir, "best_fusion.json")

    df.to_csv(csv_out, index=False)
    top10.to_csv(top10_out, index=False)

    best_payload = {
        "ms_ckpt": args.ms_ckpt,
        "tir_ckpt": args.tir_ckpt,
        "csv_path": args.csv_path,
        "data_root": args.data_root,
        "best_ms_weight": float(best_row["ms_weight"]),
        "best_tir_weight": float(best_row["tir_weight"]),
        "threshold": float(best_row["threshold"]),
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

    with open(best_json_out, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("✅ late fusion 扫描完成")
    print("=" * 80)
    print(f"Best MS weight : {best_row['ms_weight']:.2f}")
    print(f"Best TIR weight: {best_row['tir_weight']:.2f}")
    print(
        f"Best metrics   : "
        f"Acc={best_row['acc']:.4f} | "
        f"F1={best_row['f1']:.4f} | "
        f"Precision={best_row['precision']:.4f} | "
        f"Recall={best_row['recall']:.4f} | "
        f"Specificity={best_row['specificity']:.4f} | "
        f"BalancedAcc={best_row['balanced_acc']:.4f}"
    )
    print(f"Saved scan csv : {csv_out}")
    print(f"Saved top10 csv: {top10_out}")
    print(f"Saved best json: {best_json_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
