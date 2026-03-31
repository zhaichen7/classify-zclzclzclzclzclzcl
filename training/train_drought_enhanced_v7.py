"""
增强版三模态干旱分级训练 v7
增加数据增强强度，优化训练策略
"""

from models.net_drought import DroughtClassifier
from datasets.dataset_drought import build_datasets
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from tqdm import tqdm
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 增强版训练配置
# ============================================================

# 数据路径
CSV_PATH = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'

# 增强版数据增强配置
AUGMENTATION_STRENGTH = 0.9  # 增强强度从0.7提升到0.9
AUGMENTATION_FACTOR = 8      # 增强倍数从5提升到8

# 训练参数（进一步减小批次大小避免显存不足）
BATCH_SIZE = 2  # 从4减小到2
NUM_WORKERS = 1  # 进一步减少工作进程
NUM_CLASSES = 5
GRAD_ACCUM_STEPS = 4  # 梯度累积步数，等效批次大小=2×4=8

# 增强版正则化
WEIGHT_DECAY = 1e-3          # 更强的权重衰减
LABEL_SMOOTHING = 0.3        # 更强的标签平滑
GRAD_CLIP = 0.3              # 更严格的梯度裁剪

# 多阶段训练（延长训练时间）
STAGE1_EPOCHS = 30           # 第一阶段：延长到30轮
STAGE2_EPOCHS = 80           # 第二阶段：延长到80轮

SAVE_DIR = './models_enhanced'
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'drought_best_enhanced.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 显存优化配置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    torch.cuda.empty_cache()  # 清空缓存


def create_enhanced_sampler(dataset, labels):
    """增强版均衡采样器（更强的少数类别权重）"""
    # 计算每个类别的样本权重
    class_counts = np.bincount(labels)

    # 使用更强的平方根权重处理不平衡
    class_weights = 1. / np.sqrt(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 为每个样本分配权重
    sample_weights = [class_weights[label] for label in labels]

    # 创建加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


def adaptive_model_complexity(num_classes):
    """根据类别数量自适应模型复杂度"""
    if num_classes == 2:
        return 32, [2, 2], [4, 4, 4]  # 二分类：简单模型
    elif num_classes == 5:
        return 48, [3, 3], [4, 4, 4]  # 五分类：中等复杂度
    else:
        return 64, [4, 4], [8, 8, 8]  # 多分类：复杂模型


def train_one_epoch(model, loader, criterion, optimizer, device, stage, epoch, total_epochs):
    """训练一个epoch（支持梯度累积）"""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    pbar = tqdm(
        loader, desc=f'Epoch {epoch}/{total_epochs} ({stage})', leave=False)

    # 梯度累积
    optimizer.zero_grad()
    accumulation_step = 0

    for rgb, tir, ms, labels in pbar:
        rgb, tir, ms, labels = rgb.to(device), tir.to(
            device), ms.to(device), labels.to(device)

        outputs = model(rgb, tir, ms)
        loss = criterion(outputs, labels)

        # 梯度累积：损失除以累积步数
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

        accumulation_step += 1

        # 达到累积步数时更新参数
        if accumulation_step % GRAD_ACCUM_STEPS == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

    # 处理最后一个不完整的累积批次
    if accumulation_step % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        # 统计（注意：累积期间的损失需要乘以累积步数）
        running_loss += loss.item() * rgb.size(0) * GRAD_ACCUM_STEPS
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += rgb.size(0)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item() * GRAD_ACCUM_STEPS:.4f}',
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

    # 每个类别的正确预测数
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for rgb, tir, ms, labels in tqdm(loader, desc='Validating', leave=False):
            rgb, tir, ms, labels = rgb.to(device), tir.to(
                device), ms.to(device), labels.to(device)

            outputs = model(rgb, tir, ms)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * rgb.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += rgb.size(0)

            # 统计每个类别的准确率
            for i in range(NUM_CLASSES):
                class_mask = (labels == i)
                if class_mask.any():
                    class_correct[i] += (preds[class_mask]
                                         == labels[class_mask]).sum().item()
                    class_total[i] += class_mask.sum().item()

    val_loss = running_loss / total_samples
    val_acc = running_correct / total_samples

    # 打印每个类别的准确率
    print('每个类别验证准确率:')
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i] * 100
            print(f'  标签 {i}: {class_acc:.2f}%')

    return val_loss, val_acc


def main():
    print("=" * 60)
    print("增强版三模态干旱分级训练 v7")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"CSV:    {CSV_PATH}")
    print(f"Data:   {DATA_ROOT}")
    print(f"增强强度: {AUGMENTATION_STRENGTH}")
    print(f"增强倍数: {AUGMENTATION_FACTOR}")

    # 自适应模型配置
    dim, num_blocks, heads = adaptive_model_complexity(NUM_CLASSES)
    print(f"自适应模型配置: dim={dim}, blocks={num_blocks}, heads={heads}")

    # 构建数据集（增强版数据增强）
    print("\n构建数据集...")
    # 手动构建数据集以支持增强参数
    from datasets.dataset_drought import DroughtDataset
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # 读取CSV文件
    df = pd.read_csv(CSV_PATH)
    ids = df['id'].tolist()
    labels = df['label'].tolist()

    # 使用分层抽样分割训练集和验证集
    train_ids, val_ids = train_test_split(
        ids,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # 创建训练集（启用增强，设置增强参数）
    train_dataset = DroughtDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        ids=train_ids,
        augment=True,
        target_size=(224, 224),
        augmentation_factor=AUGMENTATION_FACTOR  # 增强倍数
        # 注意：数据集类目前固定使用augmentation_strength=0.7
        # 如果要修改增强强度，需要修改数据集类
    )

    # 创建验证集（不启用增强）
    val_dataset = DroughtDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        ids=val_ids,
        augment=False,
        target_size=(224, 224)
    )

    # 获取训练标签用于采样器
    train_labels = []
    for i in range(len(train_dataset)):
        _, _, _, label = train_dataset[i]
        train_labels.append(label.item() if hasattr(label, 'item') else label)

    # 创建增强版均衡采样器
    enhanced_sampler = create_enhanced_sampler(train_dataset, train_labels)

    # 数据加载器（使用增强版均衡采样器）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=enhanced_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")

    # 计算增强版类别权重
    df = pd.read_csv(CSV_PATH)
    label_counts = df['label'].value_counts().sort_index()

    # 使用平方根权重，更温和地处理不平衡
    class_weights = 1. / np.sqrt(label_counts.values)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(DEVICE)

    print('类别权重（平方根处理）:')
    for i, (label, count) in enumerate(label_counts.items()):
        print(f'  标签 {label}: {count}个样本, 权重: {class_weights[i]:.4f}')

    # 创建模型
    print("\n创建模型...")
    model = DroughtClassifier(
        dim=dim,
        num_blocks=num_blocks,
        heads=heads,
        ffn_expansion_factor=2,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # 损失函数（增强版类别权重和更强的标签平滑）
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    print("\n开始增强版训练...")

    # 第一阶段：快速收敛（高学习率）
    print(f"\n=== 第一阶段：快速收敛 ({STAGE1_EPOCHS} epochs) ===")

    optimizer_stage1 = optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 高学习率
        weight_decay=WEIGHT_DECAY
    )

    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1,
        T_max=STAGE1_EPOCHS
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience = 15  # 增强版早停耐心

    for epoch in range(1, STAGE1_EPOCHS + 1):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_stage1,
            DEVICE, 'Stage1', epoch, STAGE1_EPOCHS
        )

        # 验证
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, DEVICE)

        # 学习率调整
        scheduler_stage1.step()
        current_lr = optimizer_stage1.param_groups[0]['lr']

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch}/{STAGE1_EPOCHS}  (lr={current_lr:.2e})')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}  [{epoch_time:.1f}s]')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage1.state_dict(),
                'val_acc': val_acc
            }, BEST_MODEL_PATH)
            print('✅ Best model saved!')
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= patience:
            print(f'🚨 早停触发！{patience}个epoch验证准确率未提升')
            break

    # 第二阶段：精细调优（低学习率）
    print(f"\n=== 第二阶段：精细调优 ({STAGE2_EPOCHS} epochs) ===")

    # 加载第一阶段最佳模型
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer_stage2 = optim.AdamW(
        model.parameters(),
        lr=5e-5,  # 低学习率
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, STAGE2_EPOCHS + 1):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_stage2,
            DEVICE, 'Stage2', epoch, STAGE2_EPOCHS
        )

        # 验证
        val_loss, val_acc = validate_model(
            model, val_loader, criterion, DEVICE)

        epoch_time = time.time() - start_time

        print(f'Epoch {STAGE1_EPOCHS + epoch}/{STAGE1_EPOCHS + STAGE2_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}  |  '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}  [{epoch_time:.1f}s]')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': STAGE1_EPOCHS + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage2.state_dict(),
                'val_acc': val_acc
            }, BEST_MODEL_PATH)
            print('✅ Best model updated!')

    print("\n" + "=" * 60)
    print(f"增强版训练完成！最佳验证准确率: {best_val_acc:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
