"""
dataset_drought.py
干旱分级数据集加载器，支持 ENVI 格式（.dat + .hdr）。

数据通道说明：
  RGB (3通道)       : 可见光前3个band
  热红外 (3通道)     : 热红外前3个band
  多光谱 (8通道)     :
      多光谱5通道: NIR, Red, Blue, Green, RedEdge（各取第1个band，即第0个band）
      植被指数3通道: NDVI, GNDVI, SAVI
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from utils.advanced_augmentation import RemoteSensingAugmentation
import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    import spectral.io.envi as envi
except ImportError:
    raise ImportError("请安装 spectral 库: pip install spectral")

try:
    import cv2
except ImportError:
    raise ImportError("请安装 opencv: pip install opencv-python")


# ---------------------------------------------------------------------------
# 植被指数计算
# ---------------------------------------------------------------------------

def _safe_divide(num, denom, eps=1e-8):
    """防止除零的安全除法。"""
    return num / (denom + eps)


def compute_ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red)"""
    return _safe_divide(nir - red, nir + red)


def compute_gndvi(nir, green):
    """GNDVI = (NIR - Green) / (NIR + Green)"""
    return _safe_divide(nir - green, nir + green)


def compute_savi(nir, red, L=0.5):
    """SAVI = (1 + L) × (NIR - Red) / (NIR + Red + L)  (L=0.5 by default)"""
    return (1 + L) * _safe_divide(nir - red, nir + red + L)


# ---------------------------------------------------------------------------
# ENVI 读取辅助函数
# ---------------------------------------------------------------------------

def read_envi_band(dat_path, band_idx=0):
    """
    读取 ENVI 格式文件中的指定 band，返回 float32 的 2D numpy 数组。

    Args:
        dat_path (str): .dat 文件路径（头文件应为同名 .hdr）
        band_idx (int): 要读取的 band 索引（0-based）

    Returns:
        np.ndarray: shape (H, W), dtype float32
    """
    hdr_path = dat_path.replace('.dat', '.hdr')
    img = envi.open(hdr_path, dat_path)
    # img[:, :, band_idx] 返回 (H, W, 1) 或 (H, W)
    band = img[:, :, band_idx]
    if band.ndim == 3:
        band = band[:, :, 0]
    return band.astype(np.float32)


def read_envi_bands(dat_path, band_indices):
    """
    批量读取 ENVI 格式文件中的多个 band，返回 float32 的 3D numpy 数组。

    Args:
        dat_path (str): .dat 文件路径
        band_indices (list[int]): 要读取的 band 索引列表

    Returns:
        np.ndarray: shape (len(band_indices), H, W), dtype float32
    """
    hdr_path = dat_path.replace('.dat', '.hdr')
    img = envi.open(hdr_path, dat_path)
    bands = []
    for idx in band_indices:
        band = img[:, :, idx]
        if band.ndim == 3:
            band = band[:, :, 0]
        bands.append(band.astype(np.float32))
    return np.stack(bands, axis=0)  # (C, H, W)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def percentile_normalize(arr, low=2, high=98):
    """
    百分位数归一化，将数值映射到 [0, 1]。
    对每个通道独立处理。

    Args:
        arr (np.ndarray): shape (C, H, W)

    Returns:
        np.ndarray: shape (C, H, W), values in [0, 1]
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        lo = np.percentile(arr[c], low)
        hi = np.percentile(arr[c], high)
        if hi - lo < 1e-8:
            out[c] = np.zeros_like(arr[c])
        else:
            out[c] = np.clip((arr[c] - lo) / (hi - lo), 0.0, 1.0)
    return out


def minmax_normalize(arr):
    """
    Min-Max 归一化，将数值映射到 [0, 1]。
    对每个通道独立处理。
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        lo = arr[c].min()
        hi = arr[c].max()
        if hi - lo < 1e-8:
            out[c] = np.zeros_like(arr[c])
        else:
            out[c] = (arr[c] - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# ID → 文件名映射
# ---------------------------------------------------------------------------

def get_file_paths(sample_id, data_root):
    """
    根据样本 ID 返回各模态文件路径字典。

    文件名规则：
      ID=100:
        RGB:     {data_root}/_0519_rgb_control/_0519_20m_kejianguang_100.dat
        TIR:     {data_root}/_0519_rehongwai_control/_0519_rehongwai_20m_100.dat
        NIR:     {data_root}/_0519_nir_control/_0519_duoguangpu_20m_840_100.dat
        Red:     {data_root}/_0519_red_control/_0519_duoguangpu_20m_660_100.dat
        Green:   {data_root}/_0519_green_control/_0519_duoguangpu_20m_555_100.dat
        Blue:    {data_root}/_0519_blue_control/_0519_duoguangpu_20m_450_100.dat
        RedEdge: {data_root}/_0519_rededge_control/_0519_duoguangpu_20m_720_100.dat

    Args:
        sample_id (int): 样本编号
        data_root (str): 数据根目录

    Returns:
        dict: 各模态文件路径
    """
    sid = int(sample_id)
    return {
        'rgb': os.path.join(
            data_root, '_0519_rgb_control',
            f'_0519_20m_kejianguang_{sid}.dat'
        ),
        'tir': os.path.join(
            data_root, '_0519_rehongwai_control',
            f'_0519_rehongwai_20m_{sid}.dat'
        ),
        'nir': os.path.join(
            data_root, '_0519_nir_control',
            f'_0519_duoguangpu_20m_840_{sid}.dat'
        ),
        'red': os.path.join(
            data_root, '_0519_red_control',
            f'_0519_duoguangpu_20m_660_{sid}.dat'
        ),
        'green': os.path.join(
            data_root, '_0519_green_control',
            f'_0519_duoguangpu_20m_555_{sid}.dat'
        ),
        'blue': os.path.join(
            data_root, '_0519_blue_control',
            f'_0519_duoguangpu_20m_450_{sid}.dat'
        ),
        'rededge': os.path.join(
            data_root, '_0519_rededge_control',
            f'_0519_duoguangpu_20m_720_{sid}.dat'
        ),
    }


# ---------------------------------------------------------------------------
# Dataset 类
# ---------------------------------------------------------------------------

class DroughtDataset(Dataset):
    """
    干旱分级数据集。

    每个样本包含三组张量：
      rgb (3, H, W)  : 可见光前3个band（归一化到[0,1]）
      tir (3, H, W)  : 热红外前3个band（归一化到[0,1]）
      ms  (8, H, W)  : 多光谱5通道 + 植被指数3通道（归一化到[0,1]）
    以及对应的标签（0-4）。

    Args:
        csv_path (str): 标签 CSV 文件路径，需包含 'id' 和 'label' 列。
        data_root (str): 数据根目录。
        ids (list[int]): 要使用的样本 ID 列表。
        augment (bool): 是否进行数据增强（翻转）。
        normalize_method (str): 归一化方法，'percentile' 或 'minmax'。
    """

    def __init__(self, csv_path, data_root, ids, augment=False,
                 normalize_method='percentile', target_size=(224, 224),
                 augmentation_factor=5, modalities=['rgb', 'tir', 'ms']):
        self.data_root = data_root
        self.augment = augment
        self.normalize_method = normalize_method
        self.target_size = target_size  # (H, W)
        self.augmentation_factor = augmentation_factor
        self.modalities = modalities

        # 初始化数据增强器
        if self.augment:
            self.augmenter = RemoteSensingAugmentation(
                augmentation_strength=0.7)
        else:
            self.augmenter = None

        df = pd.read_csv(csv_path)
        df = df[df['id'].isin(ids)].reset_index(drop=True)
        self.ids = df['id'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.ids)

    def _normalize(self, arr):
        if self.normalize_method == 'percentile':
            return percentile_normalize(arr)
        return minmax_normalize(arr)

    def _augment(self, *arrays):
        """随机水平/垂直翻转，对所有数组执行相同操作。"""
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-1).copy() for a in arrays)
        if np.random.rand() > 0.5:
            arrays = tuple(np.flip(a, axis=-2).copy() for a in arrays)
        return arrays

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[idx]
        paths = get_file_paths(sample_id, self.data_root)

        H, W = self.target_size if self.target_size is not None else (224, 224)

        # RGB: 前3个band
        if 'rgb' in self.modalities:
            rgb = read_envi_bands(paths['rgb'], [0, 1, 2])           # (3, H, W)
            rgb = self._normalize(rgb)
            if self.target_size is not None:
                rgb = np.stack([cv2.resize(rgb[i], (W, H), interpolation=cv2.INTER_LINEAR)
                               for i in range(rgb.shape[0])], axis=0)
        else:
            rgb = np.zeros((3, H, W), dtype=np.float32)

        # 热红外: 前3个band
        if 'tir' in self.modalities:
            tir = read_envi_bands(paths['tir'], [0, 1, 2])           # (3, H, W)
            tir = self._normalize(tir)
            if self.target_size is not None:
                tir = np.stack([cv2.resize(tir[i], (W, H), interpolation=cv2.INTER_LINEAR)
                               for i in range(tir.shape[0])], axis=0)
        else:
            tir = np.zeros((3, H, W), dtype=np.float32)

        # 多光谱: 每个波段文件只取第0个band
        if 'ms' in self.modalities:
            nir_arr = read_envi_band(paths['nir'],     band_idx=0)  # (H, W)
            red_arr = read_envi_band(paths['red'],     band_idx=0)
            green_arr = read_envi_band(paths['green'],   band_idx=0)
            blue_arr = read_envi_band(paths['blue'],    band_idx=0)
            rededge_arr = read_envi_band(paths['rededge'], band_idx=0)

            # 植被指数
            ndvi = compute_ndvi(nir_arr, red_arr)
            gndvi = compute_gndvi(nir_arr, green_arr)
            savi = compute_savi(nir_arr, red_arr)

            # 多光谱 + VI: shape (8, H, W)
            ms = np.stack([nir_arr, red_arr, blue_arr, green_arr, rededge_arr,
                           ndvi, gndvi, savi], axis=0)
            ms = self._normalize(ms)
            if self.target_size is not None:
                ms = np.stack([cv2.resize(ms[i], (W, H), interpolation=cv2.INTER_LINEAR)
                              for i in range(ms.shape[0])], axis=0)
        else:
            ms = np.zeros((8, H, W), dtype=np.float32)

        # 数据增强 (只在所有模态都加载时或者对 TIR 加载时应用)
        if self.augment and self.augmenter is not None:
            # 简化增强，只在加载了 TIR 时进行基本的几何增强
            # 这里为了简单起见，如果开启了增强，就在 __getitem__ 里根据随机性翻转
            if np.random.rand() > 0.5:
                rgb, tir, ms = np.flip(rgb, -1).copy(), np.flip(tir, -1).copy(), np.flip(ms, -1).copy()
            if np.random.rand() > 0.5:
                rgb, tir, ms = np.flip(rgb, -2).copy(), np.flip(tir, -2).copy(), np.flip(ms, -2).copy()

        return (
            torch.from_numpy(rgb).float(),
            torch.from_numpy(tir).float(),
            torch.from_numpy(ms).float(),
            torch.tensor(label, dtype=torch.long),
        )

    def _load_sample(self, idx):
        """加载单个样本（不应用增强）"""
        sample_id = self.ids[idx]
        label = self.labels[idx]
        paths = get_file_paths(sample_id, self.data_root)

        # RGB: 前3个band
        rgb = read_envi_bands(paths['rgb'], [0, 1, 2])           # (3, H, W)

        # 热红外: 前3个band
        tir = read_envi_bands(paths['tir'], [0, 1, 2])           # (3, H, W)

        # 多光谱: 每个波段文件只取第0个band
        nir_arr = read_envi_band(paths['nir'],     band_idx=0)  # (H, W)
        red_arr = read_envi_band(paths['red'],     band_idx=0)
        green_arr = read_envi_band(paths['green'],   band_idx=0)
        blue_arr = read_envi_band(paths['blue'],    band_idx=0)
        rededge_arr = read_envi_band(paths['rededge'], band_idx=0)

        # 植被指数
        ndvi = compute_ndvi(nir_arr, red_arr)
        gndvi = compute_gndvi(nir_arr, green_arr)
        savi = compute_savi(nir_arr, red_arr)

        # 多光谱 + VI: shape (8, H, W)
        ms = np.stack([nir_arr, red_arr, blue_arr, green_arr, rededge_arr,
                       ndvi, gndvi, savi], axis=0)

        # 归一化
        rgb = self._normalize(rgb)
        tir = self._normalize(tir)
        ms = self._normalize(ms)

        return rgb, tir, ms, label


# ---------------------------------------------------------------------------
# 数据集构建函数
# ---------------------------------------------------------------------------

def build_datasets_balanced(csv_path, data_root, test_size=0.2, random_state=42,
                            augment_train=True, normalize_method='percentile', target_size=(224, 224),
                            min_samples_per_class=2, target_distribution=None, modalities=['rgb', 'tir', 'ms']):
    """
    构建训练集和验证集，手动调整验证集分布以实现均衡。

    Args:
        csv_path (str): 标签 CSV 文件路径（需含 'id' 和 'label' 列）。
        data_root (str): 数据根目录。
        test_size (float): 验证集目标比例（默认 0.2）。
        random_state (int): 随机种子（默认 42）。
        augment_train (bool): 训练集是否使用数据增强。
        normalize_method (str): 归一化方法。
        min_samples_per_class (int): 验证集中每个类别的最小样本数（默认 2）。
        target_distribution: 目标分布，如[0.2, 0.2, 0.2, 0.2, 0.2]表示每个类别20%

    Returns:
        train_dataset, val_dataset
    """
    import random
    df = pd.read_csv(csv_path)

    # 如果没有指定目标分布，使用均衡分布
    if target_distribution is None:
        target_distribution = [
            1.0/len(df['label'].unique())] * len(df['label'].unique())

    # 按标签分组
    label_groups = df.groupby('label')

    train_ids = []
    val_ids = []

    # 计算每个类别在验证集中的目标样本数
    total_val_samples = int(len(df) * test_size)

    for i, (label, group) in enumerate(label_groups):
        ids = group['id'].tolist()
        n_samples = len(ids)

        # 计算该类别在验证集中的目标样本数
        target_val_samples = int(total_val_samples * target_distribution[i])

        # 确保不超过可用样本数，且至少保留1个训练样本
        max_val_samples = min(n_samples - 1, target_val_samples)
        n_val = max(min_samples_per_class, max_val_samples)

        if n_val > 0:
            # 对该类别进行抽样
            class_train_ids, class_val_ids = train_test_split(
                ids,
                test_size=n_val/n_samples,
                random_state=random_state + i  # 不同类别使用不同随机种子
            )
            train_ids.extend(class_train_ids)
            val_ids.extend(class_val_ids)
        else:
            train_ids.extend(ids)

    # 打乱顺序
    random.shuffle(train_ids)
    random.shuffle(val_ids)

    print("Balanced split completed:")
    print(f"   Training set: {len(train_ids)} samples")
    print(f"   Validation set: {len(val_ids)} samples")

    # 统计验证集标签分布
    val_labels = df[df['id'].isin(val_ids)]['label']
    label_counts = val_labels.value_counts().sort_index()
    print("   Validation set label distribution:")
    for label, count in label_counts.items():
        percentage = count / len(val_labels) * 100
        print(f"     Label {label}: {count} samples ({percentage:.1f}%)")

    train_dataset = DroughtDataset(
        csv_path, data_root, train_ids,
        augment=augment_train,
        normalize_method=normalize_method,
        target_size=target_size,
        modalities=modalities,
    )
    val_dataset = DroughtDataset(
        csv_path, data_root, val_ids,
        augment=False,
        normalize_method=normalize_method,
        target_size=target_size,
        modalities=modalities,
    )
    return train_dataset, val_dataset


def build_datasets(csv_path, data_root, test_size=0.2, random_state=42,
                   augment_train=True, normalize_method='percentile', target_size=(224, 224),
                   balanced=True, target_distribution=None, modalities=['rgb', 'tir', 'ms']):
    """
    从 CSV 文件构建训练集和验证集，可选择使用平衡分割。

    Args:
        csv_path (str): 标签 CSV 文件路径（需含 'id' 和 'label' 列）。
        data_root (str): 数据根目录。
        test_size (float): 验证集比例（默认 0.2）。
        random_state (int): 随机种子（默认 42）。
        augment_train (bool): 训练集是否使用数据增强。
        normalize_method (str): 归一化方法。
        balanced (bool): 是否使用平衡分割（默认 True）。

    Returns:
        train_dataset, val_dataset
    """
    if balanced:
        return build_datasets_balanced(csv_path, data_root, test_size, random_state,
                                       augment_train, normalize_method, target_size,
                                       target_distribution=target_distribution,
                                       modalities=modalities)

    # 原始的分层抽样方法
    df = pd.read_csv(csv_path)
    ids = df['id'].tolist()
    labels = df['label'].tolist()

    train_ids, val_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_dataset = DroughtDataset(
        csv_path, data_root, train_ids,
        augment=augment_train,
        normalize_method=normalize_method,
        target_size=target_size,
        modalities=modalities,
    )
    val_dataset = DroughtDataset(
        csv_path, data_root, val_ids,
        augment=False,
        normalize_method=normalize_method,
        target_size=target_size,
        modalities=modalities,
    )
    return train_dataset, val_dataset


def build_dataloaders(csv_path, data_root, batch_size=8, num_workers=4,
                      test_size=0.2, random_state=42,
                      augment_train=True, normalize_method='percentile', target_size=(224, 224),
                      balanced=True, augmentation_factor=5, target_distribution=None, modalities=['rgb', 'tir', 'ms']):
    """
    构建训练/验证 DataLoader。

    Returns:
        train_loader, val_loader
    """
    train_ds, val_ds = build_datasets(
        csv_path, data_root,
        test_size=test_size,
        random_state=random_state,
        augment_train=augment_train,
        normalize_method=normalize_method,
        target_size=target_size,
        balanced=balanced,
        target_distribution=target_distribution,
        modalities=modalities,
    )

    # 如果启用增强且指定了增强倍数，创建增强数据集
    if augment_train and augmentation_factor > 0:
        from utils.advanced_augmentation import create_augmented_dataset
        train_ds = create_augmented_dataset(train_ds, augmentation_factor)
        print(
            f'Data augmentation enabled: training samples extended from {len(train_ds.original_dataset)} to {len(train_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("测试数据集加载...")
    print("=" * 60)

    csv_path = '/home/zcl/addfuse/2025label_classic5.csv'
    data_root = '/home/zcl/addfuse/dataset/'

    print(f"\nCSV 路径: {csv_path}")
    print(f"数据根目录: {data_root}")

    # 创建数据集
    print("\n创建训练集和验证集...")
    train_ds, val_ds = build_datasets(csv_path, data_root)

    print(f"✅ 训练集大小: {len(train_ds)}")
    print(f"✅ 验证集大小: {len(val_ds)}")

    # 测试加载第一个样本
    print("\n测试加载训练样本...")
    rgb, tir, ms, label = train_ds[0]

    print(f"✅ 样本加载成功！")
    print(f"  - RGB shape: {rgb.shape}")
    print(f"  - TIR shape: {tir.shape}")
    print(f"  - MS shape: {ms.shape}")
    print(f"  - Label: {label.item()}")

    # 测试加载多个样本
    print("\nTesting loading first 5 samples...")
    for i in range(min(5, len(train_ds))):
        try:
            rgb, tir, ms, label = train_ds[i]
            print(
                f"  Sample {i}: RGB{rgb.shape} TIR{tir.shape} MS{ms.shape} Label={label.item()}")
        except Exception as e:
            print(f"  Sample {i}: Loading failed - {e}")

    # 测试 DataLoader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=min(
        8, len(train_ds)), shuffle=True, num_workers=0)

    batch_rgb, batch_tir, batch_ms, batch_labels = next(iter(train_loader))
    print(f"DataLoader test successful!")
    print(f"  - Batch RGB shape: {batch_rgb.shape}")
    print(f"  - Batch TIR shape: {batch_tir.shape}")
    print(f"  - Batch MS shape: {batch_ms.shape}")
    print(f"  - Batch labels shape: {batch_labels.shape}")

    print("\n" + "=" * 60)
    print("Dataset test all passed!")
    print("=" * 60)
