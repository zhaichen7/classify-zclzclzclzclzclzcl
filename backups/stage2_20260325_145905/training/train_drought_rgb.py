"""
单模态RGB干旱分级训练
仅使用RGB图像进行干旱分级，对比三模态性能
"""

from models.net_drought_rgb import DroughtClassifierRGB
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 训练配置
# ============================================================

# 数据路径
CSV_PATH = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'

# 训练参数
BATCH_SIZE = 8  # 单模态可以使用更大的批次
NUM_WORKERS = 4
NUM_CLASSES = 5

# 正则化配置
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 0.5

# 训练轮数
EPOCHS = 100  # 单模态可能需要更多训练轮数

SAVE_DIR = './models_rgb'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_best_rgb.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_sampler(dataset, labels):
    """创建均衡采样器"""
    class_counts = np.bincount(labels)
    class_weights = 1. / np.sqrt(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


def load_rgb_data(csv_path, data_root, ids):
    """
    加载RGB数据
    仅加载RGB图像，忽略热红外和多光谱
    """
    from datasets.dataset_drought import read_envi_bands, get_file_paths

    rgb_data = []
    labels = []

    df = pd.read_csv(csv_path)
    df = df[df['id'].isin(ids)].reset_index(drop=True)

    for sample_id in tqdm(ids, desc='加载RGB数据'):
        paths = get_file_paths(sample_id, data_root)

        # 仅加载RGB图像
        rgb = read_envi_bands(paths['rgb'], [0, 1, 2])  # (3, H, W)

        rgb_data.append(rgb)

        # 获取标签
        label = df[df['id'] == sample_id]['label'].values[0]
        labels.append(label)

    return rgb_data, labels


class RGBDataset(torch.utils.data.Dataset):
    """单模态RGB数据集"""

    def __init__(self, csv_path, data_root, ids, augment=True, target_size=(224, 224)):
        self.csv_path = csv_path
        self.data_root = data_root
        self.ids = ids
        self.augment = augment
        self.target_size = target_size

        # 正确存储样本ID和标签的对应关系
        df = pd.read_csv(csv_path)
        # 确保样本ID和标签正确匹配
        self.id_to_label = {}
        for _, row in df[df['id'].isin(ids)].iterrows():
            self.id_to_label[row['id']] = row['label']

    def __len__(self):
        return len(self.ids)

    def _augment(self, rgb):
        """简单的数据增强"""
        if np.random.rand() > 0.5:
            rgb = np.flip(rgb, axis=-1).copy()
        if np.random.rand() > 0.5:
            rgb = np.flip(rgb, axis=-2).copy()
        return rgb

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        # 确保从字典中正确获取标签
        label = self.id_to_label[sample_id]

        # 实时加载RGB数据
        from datasets.dataset_drought import read_envi_bands, get_file_paths
        import cv2

        try:
            paths = get_file_paths(sample_id, self.data_root)
            rgb = read_envi_bands(paths['rgb'], [0, 1, 2])  # (3, H, W)

            # 调整图像尺寸到统一大小
            if rgb.shape[1:] != self.target_size:
                # 转置为(H, W, C)格式用于cv2.resize
                rgb_resized = np.transpose(rgb, (1, 2, 0))
                rgb_resized = cv2.resize(
                    rgb_resized, self.target_size[::-1])  # cv2使用(W, H)
                rgb = np.transpose(rgb_resized, (2, 0, 1))  # 转回(C, H, W)

            # 数据增强
            if self.augment:
                if np.random.rand() > 0.5:
                    rgb = np.flip(rgb, axis=-1).copy()
                if np.random.rand() > 0.5:
                    rgb = np.flip(rgb, axis=-2).copy()

            # 转换为张量
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return rgb_tensor, label_tensor

        except Exception as e:
            print(f"加载样本 {sample_id} 时出错: {e}")
            # 返回一个空的张量作为占位符
            rgb_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return rgb_tensor, label_tensor


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc='训练', leave=False)

    for rgb, labels in pbar:
        rgb, labels = rgb.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(rgb)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        running_loss += loss.item() * rgb.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += rgb.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{running_correct/total_samples:.2%}'
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


def validate_model(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for rgb, labels in tqdm(loader, desc='验证', leave=False):
            rgb, labels = rgb.to(device), labels.to(device)

            outputs = model(rgb)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * rgb.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += rgb.size(0)

            for i in range(NUM_CLASSES):
                class_mask = (labels == i)
                if class_mask.any():
                    class_correct[i] += (preds[class_mask]
                                         == labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()

    val_loss = running_loss / total_samples
    val_acc = running_correct / total_samples

    print('每个类别验证准确率:')
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i] * 100
            print(f'  标签 {i}: {class_acc:.2f}%')

    return val_loss, val_acc


def main():
    print("=" * 60)
    print("单模态RGB干旱分级训练")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CSV:    {CSV_PATH}")
    print(f"Data:   {DATA_ROOT}")

    # 读取数据
    print("\n加载数据...")
    df = pd.read_csv(CSV_PATH)
    ids = df['id'].tolist()
    labels = df['label'].tolist()

    # 分割训练集和验证集
    train_ids, val_ids = train_test_split(
        ids,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # 创建数据集
    train_dataset = RGBDataset(CSV_PATH, DATA_ROOT, train_ids, augment=True)
    val_dataset = RGBDataset(CSV_PATH, DATA_ROOT, val_ids, augment=False)

    # 创建采样器
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    sampler = create_sampler(train_dataset, train_labels)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    # 计算类别权重
    label_counts = df['label'].value_counts().sort_index()
    class_weights = 1. / np.sqrt(label_counts.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(DEVICE)

    print('类别权重:')
    for i, (label, count) in enumerate(label_counts.items()):
        print(f'  标签 {label}: {count}个样本, 权重: {class_weights[i]:.4f}')

    # 创建模型
    print("\n创建模型...")
    model = DroughtClassifierRGB(
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # 损失函数
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n开始训练...")

    best_val_acc = 0.0
    patience_counter = 0
    patience = 20

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)

        # 验证
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, DEVICE)

        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch}/{EPOCHS}  (lr={current_lr:.2e})')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}  [{epoch_time:.1f}s]')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, BEST_MODEL_PATH)
            print('✅ Best model saved!')
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= patience:
            print(f'🚨 早停触发！{patience}个epoch验证准确率未提升')
            break

    print("\n" + "=" * 60)
    print(f"单模态RGB训练完成！最佳验证准确率: {best_val_acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
