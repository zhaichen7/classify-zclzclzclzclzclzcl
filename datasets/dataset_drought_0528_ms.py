import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import (
    read_envi_band,
    compute_ndvi,
    compute_gndvi,
    compute_savi,
    percentile_normalize,
    minmax_normalize,
)

def get_0528_ms_paths(sample_id, data_root):
    sid = str(sample_id)
    return {
        "blue": os.path.join(
            data_root, "_0528_blue_control",
            f"_0528_duoguangpu_20m_450_transparent_mosaic_blue_warped_{sid}.hdr"
        ),
        "green": os.path.join(
            data_root, "_0528_green_control",
            f"_0528_duoguangpu_20m_555-b_transparent_mosaic_green_warped_{sid}.hdr"
        ),
        "red": os.path.join(
            data_root, "_0528_red_control",
            f"_0528_duoguangpu_20m_660-b_transparent_mosaic_red_warped_{sid}.hdr"
        ),
        "rededge": os.path.join(
            data_root, "_0528_rededge_control",
            f"_0528_duoguangpu_20m_720_transparent_mosaic_red_edge_warped_{sid}.hdr"
        ),
        "nir": os.path.join(
            data_root, "_0528_nir_control",
            f"_0528_duoguangpu_20m_840_transparent_mosaic_nir_warped_{sid}.hdr"
        ),
    }

def check_missing_0528_files(csv_path, data_root):
    df = pd.read_csv(csv_path)
    missing = []
    for _, row in df.iterrows():
        sid = row["id"]
        label = row["label"]
        paths = get_0528_ms_paths(sid, data_root)
        for k, v in paths.items():
            if not os.path.exists(v):
                missing.append((sid, label, k, v))
    return missing

class Cotton0528MSDataset(Dataset):
    def __init__(
        self,
        ids,
        labels,
        data_root,
        augment=False,
        normalize_method="percentile",
        target_size=(224, 224),
    ):
        self.ids = list(ids)
        self.labels = list(labels)
        self.data_root = data_root
        self.augment = augment
        self.normalize_method = normalize_method
        self.target_size = target_size

    def __len__(self):
        return len(self.ids)

    def _normalize(self, arr):
        if self.normalize_method == "percentile":
            return percentile_normalize(arr)
        return minmax_normalize(arr)

    def _augment(self, ms):
        if np.random.rand() > 0.5:
            ms = np.flip(ms, axis=-1).copy()
        if np.random.rand() > 0.5:
            ms = np.flip(ms, axis=-2).copy()
        return ms

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[idx]
        paths = get_0528_ms_paths(sample_id, self.data_root)

        nir_arr = read_envi_band(paths["nir"], band_idx=0)
        red_arr = read_envi_band(paths["red"], band_idx=0)
        green_arr = read_envi_band(paths["green"], band_idx=0)
        blue_arr = read_envi_band(paths["blue"], band_idx=0)
        rededge_arr = read_envi_band(paths["rededge"], band_idx=0)

        ndvi = compute_ndvi(nir_arr, red_arr)
        gndvi = compute_gndvi(nir_arr, green_arr)
        savi = compute_savi(nir_arr, red_arr)

        ms = np.stack(
            [nir_arr, red_arr, blue_arr, green_arr, rededge_arr, ndvi, gndvi, savi],
            axis=0
        ).astype(np.float32)

        ms = self._normalize(ms)

        if self.target_size is not None:
            H, W = self.target_size
            ms = np.stack(
                [cv2.resize(ms[i], (W, H), interpolation=cv2.INTER_LINEAR) for i in range(ms.shape[0])],
                axis=0
            )

        if self.augment:
            ms = self._augment(ms)

        # 保持和旧训练脚本一致：返回 rgb, tir, ms, label
        rgb = np.zeros((3, ms.shape[1], ms.shape[2]), dtype=np.float32)
        tir = np.zeros((3, ms.shape[1], ms.shape[2]), dtype=np.float32)

        return (
            torch.from_numpy(rgb).float(),
            torch.from_numpy(tir).float(),
            torch.from_numpy(ms).float(),
            torch.tensor(label, dtype=torch.long),
        )

def build_0528_balanced_split(csv_path, random_state=42, val_per_class=16):
    df = pd.read_csv(csv_path)

    train_ids, train_labels = [], []
    val_ids, val_labels = [], []

    for cls in sorted(df["label"].unique()):
        cls_df = df[df["label"] == cls].sample(frac=1, random_state=random_state)

        if len(cls_df) <= val_per_class:
            raise ValueError(
                f"class {cls} has only {len(cls_df)} samples, smaller than val_per_class={val_per_class}"
            )

        val_df = cls_df.iloc[:val_per_class]
        train_df = cls_df.iloc[val_per_class:]

        val_ids.extend(val_df["id"].tolist())
        val_labels.extend(val_df["label"].tolist())

        train_ids.extend(train_df["id"].tolist())
        train_labels.extend(train_df["label"].tolist())

    print("Balanced split completed:")
    print(f" Training set: {len(train_ids)} samples")
    print(f" Validation set: {len(val_ids)} samples")
    print(" Validation set label distribution:")
    vc = pd.Series(val_labels).value_counts().sort_index()
    for k, v in vc.items():
        print(f" Label {k}: {v} samples ({v/len(val_labels)*100:.1f}%)")

    return train_ids, train_labels, val_ids, val_labels

def build_dataloaders_0528_ms(
    csv_path,
    data_root,
    batch_size=4,
    num_workers=2,
    random_state=42,
    augment_train=True,
    normalize_method="percentile",
    target_size=(224, 224),
    augmentation_factor=0,
    val_per_class=16,
):
    train_ids, train_labels, val_ids, val_labels = build_0528_balanced_split(
        csv_path=csv_path,
        random_state=random_state,
        val_per_class=val_per_class,
    )

    train_ds = Cotton0528MSDataset(
        train_ids,
        train_labels,
        data_root=data_root,
        augment=augment_train,
        normalize_method=normalize_method,
        target_size=target_size,
    )

    val_ds = Cotton0528MSDataset(
        val_ids,
        val_labels,
        data_root=data_root,
        augment=False,
        normalize_method=normalize_method,
        target_size=target_size,
    )

    if augment_train and augmentation_factor > 0:
        from utils.advanced_augmentation import create_augmented_dataset
        train_ds = create_augmented_dataset(train_ds, augmentation_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
