import os
import sys
import random
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_datasets


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def extract_labels_from_ids(csv_path, ids):
    df = pd.read_csv(csv_path)
    id_to_label = dict(zip(df["id"].tolist(), df["label"].tolist()))
    labels = [int(id_to_label[i]) for i in ids]
    return labels


def make_targeted_sampler(labels, boost0=1.8, boost1=0.85, boost2=1.0, boost3=1.0, boost4=2.0):
    counts = np.bincount(labels, minlength=5).astype(np.float64)
    base = 1.0 / np.maximum(counts, 1.0)

    class_multiplier = np.array([boost0, boost1, boost2, boost3, boost4], dtype=np.float64)
    class_weights = base * class_multiplier

    sample_weights = np.array([class_weights[y] for y in labels], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )
    return sampler, class_weights


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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--save_dir", default="./models_ms_full8_targeted_sampler")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--boost0", type=float, default=1.8)
    parser.add_argument("--boost1", type=float, default=0.85)
    parser.add_argument("--boost2", type=float, default=1.0)
    parser.add_argument("--boost3", type=float, default=1.0)
    parser.add_argument("--boost4", type=float, default=2.0)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 72)
    print("MS full8 targeted sampler training")
    print("=" * 72)
    print("device:", device)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("lr:", args.lr)
    print("label_smoothing:", args.label_smoothing)
    print("boosts:", [args.boost0, args.boost1, args.boost2, args.boost3, args.boost4])

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

    train_ids = list(train_ds.ids)
    train_labels = extract_labels_from_ids(args.csv_path, train_ids)
    val_ids = list(val_ds.ids)
    val_labels = extract_labels_from_ids(args.csv_path, val_ids)

    print("train size:", len(train_ids))
    print("val size:", len(val_ids))
    print("train label dist:", dict(Counter(train_labels)))
    print("val label dist:", dict(Counter(val_labels)))

    sampler, class_weights = make_targeted_sampler(
        train_labels,
        boost0=args.boost0,
        boost1=args.boost1,
        boost2=args.boost2,
        boost3=args.boost3,
        boost4=args.boost4,
    )

    print("targeted sampler class weights:", class_weights.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MSClassifier(in_channels=8).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
    best_path = os.path.join(args.save_dir, "full8_targeted_best.pth")

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
            f"[Epoch {epoch:03d}] "
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
                    "boosts": [args.boost0, args.boost1, args.boost2, args.boost3, args.boost4],
                },
                best_path,
            )
            print(f"best updated: Acc={best_acc*100:.2f}%, F1={best_f1*100:.2f}%")
        else:
            patience_counter += 1
            print(f"no improve: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stop")
            break

    print("=" * 72)
    print(f"final best acc: {best_acc*100:.2f}%")
    print(f"final best f1: {best_f1*100:.2f}%")
    print(f"Saved to: {best_path}")


if __name__ == "__main__":
    main()
