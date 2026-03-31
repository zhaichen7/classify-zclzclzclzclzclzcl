import os
import sys
import random
import argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    model = MSClassifier(encoder).to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    print(f"\n📥 加载预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    loaded = False
    msg = ""
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            loaded = True
            msg = f"✅ 已加载 checkpoint: epoch={ckpt.get('epoch', 'NA')}, val_acc={ckpt.get('val_acc', 0)*100:.2f}%"
        except Exception as e:
            print(f"⚠️ strict=True 加载失败: {e}")
            try:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                loaded = True
                msg = "✅ 已用 strict=False 加载 checkpoint['model_state_dict']"
            except Exception as e2:
                print(f"⚠️ strict=False 也失败: {e2}")
    else:
        try:
            model.load_state_dict(ckpt, strict=False)
            loaded = True
            msg = "✅ 已用 strict=False 加载原始 state_dict"
        except Exception as e:
            print(f"⚠️ 原始 state_dict 加载失败: {e}")

    if not loaded:
        raise RuntimeError("无法加载预训练权重，请检查 checkpoint 格式")

    print(msg)
    return ckpt


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, grad_clip=1.0):
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

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total, 1):.2f}%"
        })

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0, desc="Val"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(loader, desc=f"Epoch {epoch} {desc}", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(ms)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            if mask.sum().item() > 0:
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / max(total, 1):.2f}%"
        })

    class_accs = []
    for i in range(5):
        if class_total[i] > 0:
            class_accs.append(class_correct[i] / class_total[i])
        else:
            class_accs.append(0.0)

    return total_loss / max(total, 1), correct / max(total, 1), class_accs


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_labels(loader):
    labels_all = []
    for _, _, _, labels in loader:
        labels_all.extend(labels.numpy().tolist())
    counter = Counter(labels_all)
    return counter


def save_ckpt(save_path, epoch, model, val_acc, class_accs, stage):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "class_accs": class_accs,
        "stage": stage
    }, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='2025label_classic5.csv')
    parser.add_argument('--data_root', default='dataset/')
    parser.add_argument('--save_dir', default='./models_ms_finetune_safe')
    parser.add_argument('--pretrain_path', default='./models_ms_opt_v1_new/drought_best.pth')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--freeze_epochs', type=int, default=8)
    parser.add_argument('--finetune_epochs', type=int, default=30)

    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--classifier_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--augmentation_factor', type=int, default=3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("🚀 MS单模态稳健微调 - 两阶段训练")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"freeze_epochs: {args.freeze_epochs}")
    print(f"finetune_epochs: {args.finetune_epochs}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"augmentation_factor: {args.augmentation_factor}")

    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        augmentation_factor=args.augmentation_factor,
        modalities=['ms']
    )
    print("✅ 数据加载完成")

    label_counter = count_labels(train_loader)
    print(f"📈 训练集标签计数: {dict(label_counter)}")

    print("\n🧠 创建模型...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    load_checkpoint(model, args.pretrain_path, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    print("\n🔍 先评估当前载入模型...")
    base_val_loss, base_val_acc, base_class_accs = evaluate(
        model, val_loader, criterion, device, epoch=0, desc="BaseEval"
    )
    print("-" * 80)
    print(f"Base val_loss={base_val_loss:.4f}, val_acc={base_val_acc * 100:.2f}%")
    print("Base class acc: " + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(base_class_accs)]))
    print("-" * 80)

    best_val_acc = base_val_acc
    best_epoch = 0
    best_stage = "base"
    best_class_accs = base_class_accs[:]

    best_path = os.path.join(args.save_dir, 'drought_best.pth')
    save_ckpt(best_path, 0, model, best_val_acc, best_class_accs, best_stage)

    # ---------------- Stage 1: 只训练分类头 ----------------
    print("\n" + "=" * 80)
    print("🎯 Stage 1: 冻结encoder，只训练分类头")
    print("=" * 80)

    freeze_encoder(model)
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.freeze_epochs, 1),
        eta_min=max(args.head_lr * 0.05, 1e-6)
    )

    for epoch in range(1, args.freeze_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch, desc="Val-S1"
        )
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"[Stage1][Epoch {epoch:03d}] lr={lr:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        print(" " * 4 + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(class_accs)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_stage = "stage1"
            best_class_accs = class_accs[:]
            save_ckpt(best_path, epoch, model, val_acc, class_accs, best_stage)
            print(f"✅ 更新最佳模型: {best_val_acc*100:.2f}% @ stage1 epoch {epoch}")

    # ---------------- Stage 2: 解冻全网小学习率微调 ----------------
    print("\n" + "=" * 80)
    print("🎯 Stage 2: 解冻全网，小学习率微调")
    print("=" * 80)

    unfreeze_all(model)
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.finetune_epochs, 1),
        eta_min=1e-6
    )

    patience_counter = 0
    for local_epoch in range(1, args.finetune_epochs + 1):
        epoch = args.freeze_epochs + local_epoch

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch, desc="Val-S2"
        )
        scheduler.step()

        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"[Stage2][Epoch {epoch:03d}] enc_lr={lrs[0]:.2e} cls_lr={lrs[1]:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        print(" " * 4 + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(class_accs)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_stage = "stage2"
            best_class_accs = class_accs[:]
            save_ckpt(best_path, epoch, model, val_acc, class_accs, best_stage)
            patience_counter = 0
            print(f"✅ 更新最佳模型: {best_val_acc*100:.2f}% @ stage2 epoch {epoch}")
        else:
            patience_counter += 1
            print(f"⏳ 未提升，patience={patience_counter}/{args.patience}")

        save_ckpt(
            os.path.join(args.save_dir, 'drought_last.pth'),
            epoch, model, val_acc, class_accs, "stage2_last"
        )

        if patience_counter >= args.patience:
            print(f"⚠️ 早停触发: 连续 {args.patience} 个 epoch 未提升")
            break

    print("\n" + "=" * 80)
    print("✅ 训练完成")
    print("=" * 80)
    print(f"最佳阶段: {best_stage}")
    print(f"最佳Epoch: {best_epoch}")
    print(f"最佳Val Acc: {best_val_acc*100:.2f}%")
    print("最佳Class Acc: " + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(best_class_accs)]))
    print(f"最佳模型已保存到: {best_path}")


if __name__ == '__main__':
    main()
