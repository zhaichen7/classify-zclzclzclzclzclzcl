import os
import sys
import json
import argparse
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def load_partial_state(module, ckpt_path, prefix):
    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        print(f"⚠️ 跳过加载 {prefix}，checkpoint 不存在: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    sub_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            sub_state[k[len(prefix):]] = v

    if not sub_state:
        print(f"⚠️ checkpoint 中未找到前缀 {prefix}，跳过: {ckpt_path}")
        return

    msg = module.load_state_dict(sub_state, strict=False)
    print(f"✅ 加载 {prefix} from {ckpt_path}")
    print(f"   missing_keys={len(msg.missing_keys)}, unexpected_keys={len(msg.unexpected_keys)}")


# ============================================================================
# Models
# ============================================================================

class TIRBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = in_features

    def forward(self, x):
        return self.backbone(x)


class FusionClassifier(nn.Module):
    def __init__(self, ms_dropout=0.3, fusion_dropout=0.4):
        super().__init__()

        self.ms_encoder = RestormerEncoder(
            inp_channels=8,
            dim=48,
            num_blocks=[4, 6],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )
        self.ms_pool = nn.AdaptiveAvgPool2d(1)

        self.tir_backbone = TIRBackbone()

        fusion_dim = 48 + self.tir_backbone.out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(ms_dropout),
            nn.Linear(64, 2)
        )

    def forward(self, tir, ms):
        ms_feat = self.ms_encoder(ms)
        ms_feat = self.ms_pool(ms_feat).view(ms_feat.size(0), -1)   # [B, 48]

        tir_feat = self.tir_backbone(tir)                           # [B, 512]

        fused = torch.cat([ms_feat, tir_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits


# ============================================================================
# Train / Eval
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, tir, ms, labels in pbar:
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(tir, ms)
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

    all_preds = []
    all_labels = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, tir, ms, labels in pbar:
        tir = tir.to(device)
        ms = ms.to(device)
        labels = labels.to(device)

        logits = model(tir, ms)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total * 100:.2f}%"
        })

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = binary_metrics_from_preds(all_preds, all_labels)

    return total_loss / total, correct / total, metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MS + TIR feature fusion binary classification")
    parser.add_argument("--csv_path", default="/home/zcl/addfuse1/2025label.csv")
    parser.add_argument("--data_root", default="/home/zcl/addfuse1/dataset")
    parser.add_argument("--save_dir", default="/home/zcl/addfuse1/two/models_ms_tir_feature_fusion_run1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fusion_dropout", type=float, default=0.4)
    parser.add_argument("--ms_dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--warmstart_ms_ckpt", default="/home/zcl/addfuse1/two/models_ms_binary_run1/binary_best.pth")
    parser.add_argument("--warmstart_tir_ckpt", default="/home/zcl/addfuse1/two/models_tir_binary_run2_conservative/binary_best.pth")
    parser.add_argument("--freeze_backbones", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    mapped_csv = os.path.join(args.save_dir, "binary_mapped.csv")
    mapping, n_rows = prepare_binary_csv(args.csv_path, mapped_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS + TIR feature fusion 二分类")
    print("=" * 80)
    print(f"Device            : {device}")
    print(f"Original csv      : {args.csv_path}")
    print(f"Mapped csv        : {mapped_csv}")
    print(f"Label map         : {mapping}")
    print(f"Rows              : {n_rows}")
    print(f"Data root         : {args.data_root}")
    print(f"Save dir          : {args.save_dir}")
    print(f"Warmstart MS ckpt : {args.warmstart_ms_ckpt}")
    print(f"Warmstart TIR ckpt: {args.warmstart_tir_ckpt}")
    print(f"Freeze backbones : {args.freeze_backbones}")
    print("=" * 80)

    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
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
    print("✅ 数据加载完成")

    print("\n🧠 创建模型...")
    model = FusionClassifier(
        ms_dropout=args.ms_dropout,
        fusion_dropout=args.fusion_dropout
    ).to(device)

    # warm start
    load_partial_state(model.ms_encoder, args.warmstart_ms_ckpt, prefix="encoder.")
    load_partial_state(model.tir_backbone.backbone, args.warmstart_tir_ckpt, prefix="backbone.")

    if args.freeze_backbones:
        for p in model.ms_encoder.parameters():
            p.requires_grad = False
        for p in model.tir_backbone.parameters():
            p.requires_grad = False
        print("✅ 已冻结 ms_encoder 与 tir_backbone，只训练 fusion_head")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型参数量: {total_params:,}")
    print(f"✅ 可训练参数量: {trainable_params:,}")

    print("\n⚖️ 计算类别权重...")
    train_labels = []
    for _, _, _, labels in train_loader:
        train_labels.extend(labels.numpy().tolist())
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)

    weights = []
    for i in range(2):
        count = label_counts.get(i, 1)
        weights.append(total_samples / (2 * count))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"✅ class_weights: {class_weights.cpu().numpy()}")

    print("\n🎯 创建损失函数...")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    print("✅ Loss: CrossEntropyLoss")

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

    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = 0
    best_metrics = None
    patience_counter = 0

    print("\n" + "=" * 80)
    print("🚀 开始训练...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, metrics = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"\n✓ Epoch {epoch:3d}/{args.epochs} | LR={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%"
        )
        print(
            "           "
            f"F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
        )

        is_better = (metrics["f1"] > best_f1) or (metrics["f1"] == best_f1 and val_acc > best_acc)

        if is_better:
            best_f1 = metrics["f1"]
            best_acc = val_acc
            best_epoch = epoch
            best_metrics = metrics
            patience_counter = 0

            ckpt_path = os.path.join(args.save_dir, "binary_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "metrics": metrics,
                "label_map": mapping,
                "config": vars(args),
            }, ckpt_path)

            with open(os.path.join(args.save_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "best_epoch": epoch,
                    "best_val_acc": val_acc,
                    "best_metrics": metrics,
                    "label_map": mapping,
                    "config": vars(args),
                }, f, ensure_ascii=False, indent=2)

            print(f"           ✅ 保存最佳模型 (epoch {epoch}, best_f1={best_f1:.4f})")
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
    print(f"最佳验证准确率: {best_acc*100:.2f}%")
    if best_metrics is not None:
        print(
            f"最佳 F1: {best_metrics['f1']:.4f} | "
            f"Precision: {best_metrics['precision']:.4f} | "
            f"Recall: {best_metrics['recall']:.4f} | "
            f"Specificity: {best_metrics['specificity']:.4f}"
        )
    print(f"结果目录: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
