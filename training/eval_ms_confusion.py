import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_datasets


class MSClassifier(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=in_channels,
            dim=48,
            num_blocks=[4, 6],
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(ckpt_path, device):
    model = MSClassifier(in_channels=8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        meta = {
            "epoch": ckpt.get("epoch", None),
            "val_acc": ckpt.get("val_acc", None),
            "val_f1": ckpt.get("val_f1", None),
        }
    else:
        model.load_state_dict(ckpt, strict=True)
        meta = {"epoch": None, "val_acc": None, "val_f1": None}

    model.eval()
    return model, meta


@torch.no_grad()
def evaluate(model, loader, ids_list, device):
    all_preds = []
    all_targets = []
    all_rows = []

    idx_ptr = 0
    for _, _, ms, labels in loader:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(ms)
        preds = logits.argmax(dim=1)

        preds_np = preds.cpu().numpy().tolist()
        labels_np = labels.cpu().numpy().tolist()

        for p, t in zip(preds_np, labels_np):
            sid = ids_list[idx_ptr] if idx_ptr < len(ids_list) else None
            all_rows.append({
                "id": sid,
                "true_label": int(t),
                "pred_label": int(p),
                "correct": int(p == t),
            })
            idx_ptr += 1

        all_preds.extend(preds_np)
        all_targets.extend(labels_np)

    return all_targets, all_preds, pd.DataFrame(all_rows)


def save_confusion(cm, save_png, save_csv):
    pd.DataFrame(cm).to_csv(save_csv, index=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, [str(i) for i in range(cm.shape[0])])
    plt.yticks(tick_marks, [str(i) for i in range(cm.shape[0])])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--ckpt_path", default="./models_ms_opt_v1_new/drought_best.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./analysis_ms_confusion")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("MS full8 confusion analysis")
    print("=" * 70)
    print("device:", device)
    print("ckpt:", args.ckpt_path)

    train_ds, val_ds = build_datasets(
        csv_path=args.csv_path,
        data_root=args.data_root,
        test_size=0.2,
        random_state=args.seed,
        augment_train=True,
        normalize_method="percentile",
        target_size=(224, 224),
        balanced=True,
        modalities=["ms"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    ids_list = list(val_ds.ids) if hasattr(val_ds, "ids") else list(range(len(val_ds)))

    model, meta = load_model(args.ckpt_path, device)
    print("checkpoint meta:", meta)

    y_true, y_pred, df_rows = evaluate(model, val_loader, ids_list, device)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0
    )
    report_txt = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    acc = (np.array(y_true) == np.array(y_pred)).mean()

    mis_df = df_rows[df_rows["correct"] == 0].copy()
    mis_df.to_csv(os.path.join(args.output_dir, "misclassified_samples.csv"), index=False)

    save_confusion(
        cm,
        os.path.join(args.output_dir, "confusion_matrix.png"),
        os.path.join(args.output_dir, "confusion_matrix.csv"),
    )

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
        f.write("\n")
        f.write(f"\nOverall accuracy: {acc*100:.2f}%\n")

    with open(os.path.join(args.output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print("")
    print("overall accuracy: %.2f%%" % (acc * 100))
    print("")
    print(report_txt)
    print("")
    print("saved files:")
    print(os.path.join(args.output_dir, "confusion_matrix.csv"))
    print(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(os.path.join(args.output_dir, "classification_report.txt"))
    print(os.path.join(args.output_dir, "misclassified_samples.csv"))


if __name__ == "__main__":
    main()
