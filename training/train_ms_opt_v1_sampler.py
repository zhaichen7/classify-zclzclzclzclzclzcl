import os
import sys
import random
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_datasets
from utils.advanced_augmentation import create_augmented_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MSClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_model(device):
    encoder = RestormerEncoder(
        inp_channels=8,
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    )
    return MSClassifier(encoder).to(device)


def extract_labels(ds):
    if hasattr(ds, 'labels'):
        return [int(x) for x in ds.labels]

    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base_labels = extract_labels(ds.dataset)
        return [int(base_labels[i]) for i in ds.indices]

    if hasattr(ds, 'datasets'):
        labels = []
        for sub_ds in ds.datasets:
            labels.extend(extract_labels(sub_ds))
        return labels

    if hasattr(ds, 'dataset'):
        try:
            return extract_labels(ds.dataset)
        except Exception:
            pass

    labels = []
    for i in range(len(ds)):
        item = ds[i]
        label = item[-1]
        if torch.is_tensor(label):
            label = int(label.item())
        labels.append(int(label))
    return labels


def build_loaders(args):
    print("\n📊 构建数据集...")
    train_ds, val_ds = build_datasets(
        csv_path=args.csv_path,
        data_root=args.data_root,
        test_size=0.2,
        random_state=args.seed,
        augment_train=True,
        normalize_method='percentile',
        target_size=(224, 224),
        balanced=True,
        modalities=['ms']
    )

    if args.augmentation_factor > 1:
        print(f"📦 扩增训练集: augmentation_factor={args.augmentation_factor}")
        train_ds = create_augmented_dataset(train_ds, args.augmentation_factor)

    train_labels = extract_labels(train_ds)
    val_labels = extract_labels(val_ds)

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)

    print(f"✅ 训练集样本数: {len(train_ds)}")
    print(f"✅ 验证集样本数: {len(val_ds)}")
    print(f"📈 训练集标签分布: {dict(sorted(train_counter.items()))}")
    print(f"📈 验证集标签分布: {dict(sorted(val_counter.items()))}")

    class_sample_count = np.array([train_counter[i] for i in range(5)], dtype=np.float64)
    class_weights = 1.0 / np.maximum(class_sample_count, 1.0)
    sample_weights = np.array([class_weights[y] for y in train_labels], dtype=np.float64)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print("✅ 已启用 WeightedRandomSampler")
    return train_loader, val_loader


def train_one_epoch(model, loader, focal_loss, ce_loss, optimizer, device, epoch, lambda_focal=0.5, lambda_ce=0.5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(ms)

        loss_focal = focal_loss(outputs, labels)
        loss_ce = ce_loss(outputs, labels)
        loss = lambda_focal * loss_focal + lambda_ce * loss_ce

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
            "acc": f"{100.0 * correct / max(total,1):.2f}%"
        })

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, ce_loss, device, epoch):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(ms)
        loss = ce_loss(outputs, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += bs

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            if mask.sum().item() > 0:
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total,1):.2f}%"
        })

    class_accs = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(5)
    ]
    return total_loss / max(total, 1), correct / max(total, 1), class_accs


def save_ckpt(save_path, epoch, model, val_acc, class_accs):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "class_accs": class_accs
    }, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='2025label_classic5.csv')
    parser.add_argument('--data_root', default='dataset/')
    parser.add_argument('--save_dir', default='./models_ms_opt_v1_sampler')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--lambda_focal', type=float, default=0.5)
    parser.add_argument('--lambda_ce', type=float, default=0.5)
    parser.add_argument('--augmentation_factor', type=int, default=2)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS单模态冲分版：optimized_v1 + WeightedRandomSampler")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"gamma: {args.gamma}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"lambda_focal: {args.lambda_focal}")
    print(f"lambda_ce: {args.lambda_ce}")
    print(f"augmentation_factor: {args.augmentation_factor}")
    print(f"seed: {args.seed}")

    train_loader, val_loader = build_loaders(args)

    print("\n🧠 创建模型...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    focal_loss = FocalLoss(gamma=args.gamma)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_class_accs = [0.0] * 5
    best_path = os.path.join(args.save_dir, 'drought_best.pth')
    patience_counter = 0

    print("\n🚀 开始训练...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, focal_loss, ce_loss, optimizer, device, epoch,
            lambda_focal=args.lambda_focal, lambda_ce=args.lambda_ce
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, ce_loss, device, epoch
        )
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch:03d}] lr={lr:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        print(" " * 4 + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(class_accs)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_class_accs = class_accs[:]
            save_ckpt(best_path, epoch, model, val_acc, class_accs)
            patience_counter = 0
            print(f"✅ 更新最佳模型: {best_val_acc*100:.2f}% @ epoch {epoch}")
        else:
            patience_counter += 1
            print(f"⏳ 未提升，patience={patience_counter}/{args.patience}")

        save_ckpt(
            os.path.join(args.save_dir, 'drought_last.pth'),
            epoch, model, val_acc, class_accs
        )

        if patience_counter >= args.patience:
            print(f"⚠️ 早停触发: 连续 {args.patience} 个 epoch 未提升")
            break

    print("\n" + "=" * 80)
    print("✅ 训练完成")
    print("=" * 80)
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳Val Acc: {best_val_acc*100:.2f}%")
    print("最佳Class Acc: " + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(best_class_accs)]))
    print(f"最佳模型已保存到: {best_path}")


if __name__ == '__main__':
    main()
