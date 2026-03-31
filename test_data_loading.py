"""
测试数据加载是否正确
验证单模态RGB数据读取和标签匹配
"""

from datasets.dataset_drought import read_envi_bands, get_file_paths
import os
import sys
import pandas as pd


# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据路径
CSV_PATH = '/home/zcl/addfuse/2025label_classic5.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试数据加载")
    print("=" * 60)

    # 读取CSV文件
    df = pd.read_csv(CSV_PATH)
    print(f"CSV文件总样本数: {len(df)}")
    print(f"标签分布:")
    print(df['label'].value_counts().sort_index())

    # 测试前5个样本
    test_ids = df['id'].head(5).tolist()

    print(f"\n测试前5个样本:")
    for i, sample_id in enumerate(test_ids):
        print(f"\n样本 {i+1}: ID={sample_id}")

        # 获取标签
        label = df[df['id'] == sample_id]['label'].values[0]
        print(f"  标签: {label}")

        try:
            # 获取文件路径
            paths = get_file_paths(sample_id, DATA_ROOT)
            print(f"  文件路径: {paths}")

            # 加载RGB数据
            rgb = read_envi_bands(paths['rgb'], [0, 1, 2])
            print(f"  RGB形状: {rgb.shape}")
            print(f"  RGB数据范围: [{rgb.min():.3f}, {rgb.max():.3f}]")

        except Exception as e:
            print(f"  错误: {e}")

    # 检查所有样本的文件是否存在
    print(f"\n检查所有样本的文件存在性...")
    missing_files = []

    for sample_id in df['id'].tolist():
        try:
            paths = get_file_paths(sample_id, DATA_ROOT)
            # 检查每个文件是否存在
            for file_type, file_path in paths.items():
                if not os.path.exists(file_path):
                    missing_files.append((sample_id, file_type, file_path))
        except Exception as e:
            missing_files.append((sample_id, 'all', str(e)))

    if missing_files:
        print(f"发现 {len(missing_files)} 个缺失文件:")
        for sample_id, file_type, file_path in missing_files[:5]:  # 只显示前5个
            print(f"  样本 {sample_id}: {file_type} 文件缺失 - {file_path}")
        if len(missing_files) > 5:
            print(f"  ... 还有 {len(missing_files) - 5} 个缺失文件")
    else:
        print("所有样本的文件都存在！")

    # 检查标签分布
    print(f"\n标签分布详细分析:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  标签 {label}: {count}个样本 ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("数据加载测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_data_loading()
