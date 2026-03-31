"""
数据预处理脚本
将ENVI文件预处理为numpy数组，加速训练
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from dataset_drought import read_envi_file, normalize_percentile


def preprocess_all_data(csv_path, data_root, save_dir='./preprocessed_data', target_size=(224, 224)):
    """预处理所有数据"""

    os.makedirs(save_dir, exist_ok=True)

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 存储预处理后的数据
    preprocessed_data = {}

    print(f'开始预处理 {len(df)} 个样本...')

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row['id']
        label = row['label']

        try:
            # 读取RGB数据
            rgb_path = os.path.join(data_root, f'{sample_id}_rgb')
            rgb_data = read_envi_file(rgb_path)
            rgb_data = rgb_data[:3]  # 取前3个通道

            # 读取热红外数据
            tir_path = os.path.join(data_root, f'{sample_id}_tir')
            tir_data = read_envi_file(tir_path)
            tir_data = tir_data[:3]  # 取前3个通道

            # 读取多光谱数据
            ms_path = os.path.join(data_root, f'{sample_id}_ms')
            ms_data = read_envi_file(ms_path)
            ms_data = ms_data[:1]  # 取第1个通道

            # 归一化
            rgb_norm = normalize_percentile(rgb_data)
            tir_norm = normalize_percentile(tir_data)
            ms_norm = normalize_percentile(ms_data)

            # 存储预处理后的数据
            preprocessed_data[sample_id] = {
                'rgb': rgb_norm.astype(np.float32),
                'tir': tir_norm.astype(np.float32),
                'ms': ms_norm.astype(np.float32),
                'label': label
            }

        except Exception as e:
            print(f'预处理样本 {sample_id} 失败: {e}')
            continue

    # 保存预处理数据
    save_path = os.path.join(save_dir, 'preprocessed_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    print(f'预处理完成！数据已保存至: {save_path}')
    print(f'成功预处理 {len(preprocessed_data)} 个样本')

    return preprocessed_data


def load_preprocessed_data(save_path='./preprocessed_data/preprocessed_data.pkl'):
    """加载预处理数据"""

    if not os.path.exists(save_path):
        raise FileNotFoundError(f'预处理数据文件不存在: {save_path}')

    with open(save_path, 'rb') as f:
        preprocessed_data = pickle.load(f)

    print(f'加载预处理数据: {len(preprocessed_data)} 个样本')
    return preprocessed_data


if __name__ == '__main__':
    # 预处理所有数据
    preprocessed_data = preprocess_all_data(
        csv_path='/home/zcl/addfuse/2025label_classic5.csv',
        data_root='/home/zcl/addfuse/dataset/',
        save_dir='./preprocessed_data'
    )
