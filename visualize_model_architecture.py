import matplotlib.pyplot as plt


def draw_beautiful_network():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    # 配置节点颜色
    color_inputs = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    color_encoder = '#a0c4ff'
    color_fuse = '#ffd6a5'
    color_gap = '#bdb2ff'
    color_classify = '#ffc300'
    color_output = '#00b894'

    # 输入分支
    for i, txt in enumerate(['RGB输入\n(3通道)', '热红外输入\n(3通道)', '多光谱输入\n(8通道)']):
        ax.add_patch(plt.Rectangle((i*3, 7), 2, 1,
                     color=color_inputs[i], alpha=0.8, ec='black', lw=2))
        ax.text(i*3+1, 7.5, txt, fontsize=14, ha='center',
                va='center', weight='bold', color='white')
        ax.arrow(i*3+1, 7, i*3+1, 6.3, head_width=0.3,
                 head_length=0.2, fc=color_inputs[i], ec='black', lw=2)

    # Restormer编码器
    for i, txt in enumerate(['Restormer编码器',]*3):
        ax.add_patch(plt.Rectangle((i*3, 6), 2, 1,
                     color=color_encoder, alpha=0.8, ec='black', lw=2))
        ax.text(i*3+1, 6.5, txt, fontsize=14,
                ha='center', va='center', weight='bold')

    # 下箭头至融合层
    for i in range(3):
        ax.arrow(i*3+1, 6, 4.5, 5.5, head_width=0.3,
                 head_length=0.2, fc=color_fuse, ec='black', lw=2)

    # 特征融合层
    ax.add_patch(plt.Rectangle((3, 5), 3, 1, color=color_fuse,
                 alpha=0.85, ec='black', lw=2))
    ax.text(4.5, 5.5, '特征拼接与融合层\n(通道拼接)', fontsize=15,
            ha='center', va='center', weight='bold', color='black')
    ax.arrow(4.5, 5, 4.5, 4.3, head_width=0.3,
             head_length=0.2, fc=color_gap, ec='black', lw=2)

    # GAP池化
    ax.add_patch(plt.Rectangle((3, 4), 3, 1, color=color_gap,
                 alpha=0.9, ec='black', lw=2))
    ax.text(4.5, 4.5, '全局平均池化\nGAP', fontsize=14, ha='center',
            va='center', weight='bold', color='black')
    ax.arrow(4.5, 4, 4.5, 3.3, head_width=0.3, head_length=0.2,
             fc=color_classify, ec='black', lw=2)

    # 分类头
    ax.add_patch(plt.Rectangle((3, 3), 3, 1, color=color_classify,
                 alpha=0.9, ec='black', lw=2))
    ax.text(4.5, 3.5, '分类头\n192→256→5', fontsize=14, ha='center',
            va='center', weight='bold', color='black')
    ax.arrow(4.5, 3, 4.5, 2.3, head_width=0.2, head_length=0.2,
             fc=color_output, ec='black', lw=2)

    # 输出
    ax.add_patch(plt.Rectangle((3, 2), 3, 1, color=color_output,
                 alpha=0.9, ec='black', lw=2))
    ax.text(4.5, 2.5, '输出: 5类干旱等级\n(Level 0-4)', fontsize=15,
            ha='center', va='center', weight='bold', color='white')

    plt.tight_layout()
    plt.savefig('beautified_drought_architecture.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_beautiful_network()
