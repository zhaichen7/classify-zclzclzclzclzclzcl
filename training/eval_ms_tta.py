import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_datasets


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


def load_checkpoint(model, ckpt_path, device):
    print(f"\n📥 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(f"✅ 已加载 checkpoint: epoch={ckpt.get('epoch', 'NA')}, val_acc={ckpt.get('val_acc', 0)*100:.2f}%")
    else:
        model.load_state_dict(ckpt, strict=True)
        print("✅ 已加载原始 state_dict")


def build_val_loader(args):
    print("\n📊 构建验证集...")
    _, val_ds = build_datasets(
        csv_path=args.csv_path,
        data_root=args.data_root,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        normalize_method='percentile',
        target_size=(224, 224),
        balanced=True,
        modalities=['ms']
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"✅ 验证集样本数: {len(val_ds)}")
    return val_loader


def tta_logits(model, ms):
    # 4-view TTA
    views = [
        ms,
        torch.flip(ms, dims=[3]),       # 水平翻转
        torch.flip(ms, dims=[2]),       # 垂直翻转
        torch.flip(ms, dims=[2, 3])     # 水平+垂直翻转
    ]

    logits_sum = None
    for v in views:
        logits = model(v)
        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = logits_sum + logits

    return logits_sum / len(views)


@torch.no_grad()
def evaluate_plain(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(loader, desc="Plain Eval", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(ms)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            if mask.sum().item() > 0:
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({"acc": f"{100.0 * correct / max(total,1):.2f}%"})

    class_accs = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(5)
    ]
    return correct / max(total, 1), class_accs


@torch.no_grad()
def evaluate_tta(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5

    pbar = tqdm(loader, desc="TTA Eval", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = tta_logits(model, ms)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            if mask.sum().item() > 0:
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()

        pbar.set_postfix({"acc": f"{100.0 * correct / max(total,1):.2f}%"})

    class_accs = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(5)
    ]
    return correct / max(total, 1), class_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='2025label_classic5.csv')
    parser.add_argument('--data_root', default='dataset/')
    parser.add_argument('--ckpt_path', default='./models_ms_opt_v1_new/drought_best.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🚀 MS单模态 TTA 评估")
    print("=" * 80)
    print(f"Device: {device}")

    val_loader = build_val_loader(args)

    print("\n🧠 创建模型...")
    model = build_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")

    load_checkpoint(model, args.ckpt_path, device)

    print("\n🔍 普通评估...")
    plain_acc, plain_class_accs = evaluate_plain(model, val_loader, device)
    print("-" * 80)
    print(f"Plain val_acc = {plain_acc * 100:.2f}%")
    print("Plain class acc: " + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(plain_class_accs)]))

    print("\n🔍 TTA评估...")
    tta_acc, tta_class_accs = evaluate_tta(model, val_loader, device)
    print("-" * 80)
    print(f"TTA val_acc = {tta_acc * 100:.2f}%")
    print("TTA class acc: " + " | ".join([f"c{i}={a*100:.1f}%" for i, a in enumerate(tta_class_accs)]))

    delta = (tta_acc - plain_acc) * 100
    print("-" * 80)
    print(f"TTA 相对 Plain 变化: {delta:+.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
