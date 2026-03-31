"""
轻量版二分类干旱检测
- 减小模型 (dim=48)
- 减小 batch size (4)
- 梯度累积
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

from net_drought import DroughtClassifier
from dataset_drought import DroughtDataset

CSV_PATH = '/home/zcl/addfuse/2025label_binary.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'

# ========== 显存优化配置 ==========
BATCH_SIZE = 4  # 减小
GRADIENT_ACCUMULATION = 2  # 梯度累积
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_DIR = './models_binary'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_binary_best.pth')


def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (rgb, tir, ms, labels) in enumerate(pbar):
        rgb, tir, ms, labels = rgb.to(device), tir.to(
            device), ms.to(device), labels.to(device)

        if labels.dtype != torch.long:
            labels = labels.long()

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item()*accum_steps:.4f}',
                         'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    pbar = tqdm(loader, desc="Validation", leave=False)
    for rgb, tir, ms, labels in pbar:
        rgb, tir, ms, labels = rgb.to(device), tir.to(
            device), ms.to(device), labels.to(device)

        if labels.dtype != torch.long:
            labels = labels.long()

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

    print(f"\n  各类别准确率:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            class_name = "无旱" if i == 0 else "有旱"
            print(
                f"    Class {i} ({class_name}): {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    # F1 分数
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary')
    print(f"\n  精确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1:.4f}")

    return total_loss / len(loader), 100. * correct / total, f1


def main():
    print('=' * 60)
    print('轻量版二分类训练')
    print('=' * 60)

    df = pd.read_csv(CSV_PATH)
    all_ids = df['id'].tolist()
    all_labels = df['label'].tolist()

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        all_ids, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"\n数据集分布:")
    label_counts = Counter(all_labels)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percent = count / len(all_labels) * 100
        label_name = "无旱" if label == 0 else "有旱"
        print(
            f"  Class {label} ({label_name}): {count:3d} 个 ({percent:5.1f}%)")

    # 减少数据增强
    from advanced_augmentation import create_augmented_dataset

    base_train_dataset = DroughtDataset(
        csv_path=CSV_PATH, data_root=DATA_ROOT, ids=train_ids,
        augment=True, normalize_method='percentile', target_size=(224, 224)
    )

    train_dataset = create_augmented_dataset(
        base_train_dataset, augmentation_factor=2)  # 3倍

    val_dataset = DroughtDataset(
        csv_path=CSV_PATH, data_root=DATA_ROOT, ids=val_ids,
        augment=False, normalize_method='percentile', target_size=(224, 224)
    )

    print(f'\n训练样本: {len(base_train_dataset)} → {len(train_dataset)} (3x增强)')
    print(f'验证样本: {len(val_dataset)}')
    print(
        f'有效 batch size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 模型 (减小到 dim=48)
    model = DroughtClassifier(
        dim=48,  # 从 64 降到 48
        num_blocks=[3, 3],
        heads=[6, 6, 6],
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # 类别权重
    class_weights = []
    for i in range(NUM_CLASSES):
        count = label_counts[i]
        weight = len(all_labels) / (NUM_CLASSES * count)
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    print(f"\n类别权重:")
    for i, w in enumerate(class_weights):
        label_name = "无旱" if i == 0 else "有旱"
        print(f"  Class {i} ({label_name}): {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_f1 = 0.0
    patience = 20
    patience_counter = 0

    print('\n开始训练...\n')

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch {epoch}/{EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, GRADIENT_ACCUMULATION)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}')

        if val_acc > best_val_acc or (val_acc >= best_val_acc - 1 and val_f1 > best_f1):
            best_val_acc = max(val_acc, best_val_acc)
            best_f1 = max(val_f1, best_f1)
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_f1': val_f1,
            }, BEST_MODEL_PATH)
            print(f'✅ Best! Val Acc: {val_acc:.2f}%, F1: {val_f1:.4f}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\n早停！')
            break
        print()

    print(f'\n训练完成！')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'最佳 F1 分数: {best_f1:.4f}')


if __name__ == '__main__':
    main()
