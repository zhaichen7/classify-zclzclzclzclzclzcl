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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import DroughtDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MSClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8,
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


def make_loader(csv_path, data_root, ids, batch_size, num_workers, augment):
    ds = DroughtDataset(
        csv_path=csv_path,
        data_root=data_root,
        ids=ids,
        augment=augment,
        normalize_method="percentile",
        target_size=(224, 224),
        modalities=["ms"],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def class_weights_from_labels(labels, device):
    counts = np.bincount(labels, minlength=5).astype(np.float32)
    weights = len(labels) / (5.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
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
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} Val", leave=True)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_kfold_restormer_ok")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"MS Restormer KFold (K={args.num_folds})")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs per fold: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Label smoothing: {args.label_smoothing}")
    print("=" * 70)

    df = pd.read_csv(args.csv_path)
    labels = df["label"].astype(int).values

    skf = StratifiedKFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print("\n" + "=" * 70)
        print(f"Fold {fold}/{args.num_folds}")
        print("=" * 70)

        train_ids = df.iloc[train_idx]["id"].tolist()
        val_ids = df.iloc[val_idx]["id"].tolist()
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        print(f"Train size: {len(train_ids)}")
        print(f"Val size: {len(val_ids)}")
        print(f"Train label dist: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
        print(f"Val label dist: {dict(zip(*np.unique(val_labels, return_counts=True)))}")

        train_loader = make_loader(
            args.csv_path, args.data_root, train_ids,
            args.batch_size, args.num_workers, True
        )
        val_loader = make_loader(
            args.csv_path, args.data_root, val_ids,
            args.batch_size, args.num_workers, False
        )

        model = MSClassifier().to(device)
        weights = class_weights_from_labels(train_labels, device)
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

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_loss, val_acc, val_f1 = evaluate(
                model, val_loader, criterion, device, epoch
            )
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[Fold {fold}][Epoch {epoch:03d}] "
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
                        "fold": fold,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                    },
                    os.path.join(args.save_dir, f"fold{fold}_best.pth"),
                )
                print(f"Fold {fold} best updated: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%")
            else:
                patience_counter += 1
                print(f"Fold {fold} no improve: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"Early stop on fold {fold}")
                break

        fold_results.append({
            "fold": fold,
            "acc": best_acc,
            "f1": best_f1,
        })
        print(f"Fold {fold} best acc: {best_acc*100:.2f}%, F1: {best_f1*100:.2f}%")

    accs = [r["acc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]

    print("\n" + "=" * 70)
    print("KFold final results")
    print("=" * 70)
    for r in fold_results:
        print(f"Fold {r['fold']}: Acc={r['acc']*100:.2f}%, F1={r['f1']*100:.2f}%")

    print("")
    print(f"Acc mean +- std = {np.mean(accs)*100:.2f}% +- {np.std(accs)*100:.2f}%")
    print(f"F1  mean +- std = {np.mean(f1s)*100:.2f}% +- {np.std(f1s)*100:.2f}%")
    print("=" * 70)
    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
