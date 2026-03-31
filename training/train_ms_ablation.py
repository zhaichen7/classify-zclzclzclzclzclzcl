import os
import sys
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_ms_channels(ms, variant):
    if variant == "full8":
        return ms
    if variant == "raw5":
        return ms[:, :5, :, :]
    if variant == "vi3":
        return ms[:, 5:, :, :]
    raise ValueError(f"Unknown variant: {variant}")


def get_in_channels(variant):
    if variant == "full8":
        return 8
    if variant == "raw5":
        return 5
    if variant == "vi3":
        return 3
    raise ValueError(f"Unknown variant: {variant}")


class MSClassifier(nn.Module):
    def __init__(self, in_channels):
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


def class_weights_from_loader(loader, device):
    labels_all = []
    for _, _, _, labels in loader:
        labels_all.extend(labels.numpy().tolist())
    counts = np.bincount(labels_all, minlength=5).astype(np.float32)
    weights = len(labels_all) / (5.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, variant):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
        ms = select_ms_channels(ms, variant).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(ms)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total, 1):.2f}%"
        })

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, variant):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, _, ms, labels in pbar:
        ms = select_ms_channels(ms, variant).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(ms)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total, 1):.2f}%"
        })

    acc = correct / max(total, 1)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return total_loss / max(total, 1), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_ablation")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ms_variant", choices=["full8", "raw5", "vi3"], default="full8")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"MS ablation: {args.ms_variant}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Label smoothing: {args.label_smoothing}")

    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=args.seed,
        augment_train=True,
        balanced=True,
        augmentation_factor=1,
        modalities=["ms"]
    )

    in_channels = get_in_channels(args.ms_variant)
    model = MSClassifier(in_channels=in_channels).to(device)
    weights = class_weights_from_loader(train_loader, device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )

    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    best_path = os.path.join(args.save_dir, f"{args.ms_variant}_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.ms_variant
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, epoch, args.ms_variant
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{args.ms_variant}][Epoch {epoch:03d}] "
            f"lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc*100:.2f}% "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc*100:.2f}% "
            f"val_f1={val_f1*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "ms_variant": args.ms_variant,
                },
                best_path,
            )
            print(f"{args.ms_variant} best updated: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"{args.ms_variant} no improve: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"Early stop on {args.ms_variant}")
            break

    print("=" * 70)
    print(f"{args.ms_variant} final best acc: {best_acc*100:.2f}%")
    print(f"{args.ms_variant} final best f1: {best_f1*100:.2f}%")
    print(f"Saved to: {best_path}")


if __name__ == "__main__":
    main()
