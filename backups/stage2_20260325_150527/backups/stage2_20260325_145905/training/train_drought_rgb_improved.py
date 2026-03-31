"""
改进的单模态RGB干旱分级训练
解决类别不平衡和训练不足问题
"""

from datasets.dataset_drought import read_envi_bands, get_file_paths
from models.net_drought_rgb import DroughtClassifierRGB
from sklearn.model_selection import train_test_split
import cv2
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 命令行参数
parser = argparse.ArgumentParser(description='改进的单模态RGB干旱分级训练')
parser.add_argument('--csv_path', type=str,
                    default='/home/zcl/addfuse/2025label_classic5.csv')
parser.add_argument('--data_root', type=str,
                    default='/home/zcl/addfuse/dataset/')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=30)
args = parser.parse_args()

CSV_PATH = args.csv_path
DATA_ROOT = args.data_root
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
PATIENCE = args.patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = './models_rgb_improved'

os.makedirs(MODEL_DIR, exist_ok=True)


class ImprovedRGBDataset(torch.utils.data.Dataset):
    """改进的单模态RGB数据集"""

    def __init__(self, csv_path, data_root, ids, augment=True, target_size=(224, 224)):
        self.csv_path = csv_path
        self.data_root = data_root
        self.ids = ids
        self.augment = augment
        self.target_size = target_size

        # 正确存储样本ID和标签的对应关系
        df = pd.read_csv(csv_path)
        self.id_to_label = {}
        for _, row in df[df['id'].isin(ids)].iterrows():
            self.id_to_label[row['id']] = row['label']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.id_to_label[sample_id]

        try:
            paths = get_file_paths(sample_id, self.data_root)
            rgb = read_envi_bands(paths['rgb'], [0, 1, 2])  # (3, H, W)

            # 调整图像尺寸到统一大小
            if rgb.shape[1:] != self.target_size:
                rgb_resized = np.transpose(rgb, (1, 2, 0))
                rgb_resized = cv2.resize(rgb_resized, self.target_size[::-1])
                rgb = np.transpose(rgb_resized, (2, 0, 1))

            # 增强的数据增强
            if self.augment:
                # 水平翻转
                if np.random.rand() > 0.5:
                    rgb = np.flip(rgb, axis=-1).copy()
                # 垂直翻转
                if np.random.rand() > 0.5:
                    rgb = np.flip(rgb, axis=-2).copy()
                # 随机旋转90度
                if np.random.rand() > 0.5:
                    k = np.random.randint(1, 4)
                    rgb = np.rot90(rgb, k, axes=(1, 2)).copy()
                # 随机亮度调整
                if np.random.rand() > 0.5:
                    brightness = np.random.uniform(0.8, 1.2)
                    rgb = np.clip(rgb * brightness, 0, 255)

            # 转换为张量并归一化
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32) / 255.0
            label_tensor = torch.tensor(label, dtype=torch.long)

            return rgb_tensor, label_tensor

        except Exception as e:
            print(f"加载样本 {sample_id} 时出错: {e}")
            # 返回一个随机的RGB图像作为占位符
            rgb_tensor = torch.rand((3, 224, 224), dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return rgb_tensor, label_tensor


def calculate_class_weights(labels):
    """计算类别权重以解决不平衡问题"""
    from collections import Counter
    class_counts = Counter(labels)
    total_samples = len(labels)

    # 使用逆频率权重
    weights = {}
    for label, count in class_counts.items():
        weights[label] = total_samples / (len(class_counts) * count)

    # 转换为张量
    weight_tensor = torch.tensor([weights[i]
                                 for i in range(5)], dtype=torch.float32)
    return weight_tensor


def train():
    print("=" * 60)
    print("改进的单模态RGB干旱分级训练")
    print("=" * 60)

    # 加载数据
    df = pd.read_csv(CSV_PATH)
    all_ids = df['id'].tolist()
    all_labels = df['label'].tolist()

    print(f"总样本数: {len(all_ids)}")
    print(f"标签分布: {dict(df['label'].value_counts().sort_index())}")

    # 计算类别权重
    class_weights = calculate_class_weights(all_labels)
    print(f"类别权重: {class_weights.tolist()}")

    # 分割训练集和验证集
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2,
                                          random_state=42,
                                          stratify=all_labels)

    print(f"训练集: {len(train_ids)} 个样本")
    print(f"验证集: {len(val_ids)} 个样本")

    # 创建数据集
    train_dataset = ImprovedRGBDataset(
        CSV_PATH, DATA_ROOT, train_ids, augment=True)
    val_dataset = ImprovedRGBDataset(
        CSV_PATH, DATA_ROOT, val_ids, augment=False)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = DroughtClassifierRGB(
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        num_classes=5,
    ).to(DEVICE)

    # 使用加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     patience=10, factor=0.5)

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    patience_counter = 0

    print("\n开始训练...")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        for rgb, labels in pbar:
            rgb, labels = rgb.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })

        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for rgb, labels in val_loader:
                rgb, labels = rgb.to(DEVICE), labels.to(DEVICE)
                outputs = model(rgb)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        # 学习率调度
        scheduler.step(val_acc)

        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  训练 Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}%')
        print(f'  验证 Loss: {val_losses[-1]:.4f}, Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }

            torch.save(checkpoint, os.path.join(
                MODEL_DIR, 'drought_best_rgb_improved.pth'))
            print(f'✅ 最佳模型保存! (验证准确率: {val_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'  早停计数器: {patience_counter}/{PATIENCE}')

        # 早停检查
        if patience_counter >= PATIENCE:
            print(f'\n🚨 早停触发! 最佳验证准确率: {best_val_acc:.2f}%')
            break

        print('-' * 50)

    print(f"\n训练完成! 最终最佳验证准确率: {best_val_acc:.2f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('损失曲线')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('准确率曲线')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存到: {MODEL_DIR}/training_curves.png")
    print("=" * 60)


if __name__ == "__main__":
    train()
