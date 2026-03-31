"""绘制训练曲线"""

import matplotlib.pyplot as plt
import re
import matplotlib
matplotlib.use('Agg')

LOG_FILE = '../train_best.log'
SAVE_DIR = './results'

epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

print("解析训练日志...")

with open(LOG_FILE, 'r') as f:
    for line in f:
        # 匹配: Train Loss: 1.5430, Acc: 33.93%  |  Val Loss: 1.4908, Acc: 34.12%
        match = re.search(
            r'Train Loss: ([\d.]+), Acc: ([\d.]+)%.*Val Loss: ([\d.]+), Acc: ([\d.]+)%', line)
        if match:
            epochs.append(len(epochs) + 1)
            train_loss.append(float(match.group(1)))
            train_acc.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            val_acc.append(float(match.group(4)))

print(f"解析到 {len(epochs)} 个 epoch")

if len(epochs) > 0:
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练损失
    axes[0, 0].plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # 2. 验证损失
    axes[0, 1].plot(epochs, val_loss, 'r-', linewidth=2, label='Val Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # 3. 训练准确率
    axes[1, 0].plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # 4. 验证准确率
    axes[1, 1].plot(epochs, val_acc, 'r-', linewidth=2, label='Val Acc')
    axes[1, 1].axhline(y=max(val_acc), color='g', linestyle='--',
                       label=f'Best: {max(val_acc):.2f}%')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/training_curves.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {SAVE_DIR}/training_curves.png")

    # 统计信息
    print(f"\n训练统计:")
    print(
        f"  最佳验证准确率: {max(val_acc):.2f}% (Epoch {val_acc.index(max(val_acc)) + 1})")
    print(f"  最终训练准确率: {train_acc[-1]:.2f}%")
    print(f"  最终验证准确率: {val_acc[-1]:.2f}%")
    print(f"  最低验证损失: {min(val_loss):.4f}")
else:
    print("❌ 未找到训练数据")
