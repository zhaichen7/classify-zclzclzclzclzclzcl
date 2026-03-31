import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.dataset_drought import DroughtDataset


def to_hwc(x):
    return np.transpose(x, (1, 2, 0))


def norm01(img):
    img = np.asarray(img, dtype=np.float32)
    mn = np.nanmin(img)
    mx = np.nanmax(img)
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def make_rgb_like(ms):
    # ms order:
    # 0 nir, 1 red, 2 blue, 3 green, 4 rededge, 5 ndvi, 6 gndvi, 7 savi
    rgb = np.stack([ms[1], ms[3], ms[2]], axis=0)
    return norm01(to_hwc(rgb))


def make_false_color(ms):
    # false color: NIR, Red, Green
    fc = np.stack([ms[0], ms[1], ms[3]], axis=0)
    return norm01(to_hwc(fc))


def save_sample_figure(ms, sid, true_label, pred_label, out_path):
    rgb_like = make_rgb_like(ms)
    false_color = make_false_color(ms)
    ndvi = norm01(ms[5])
    gndvi = norm01(ms[6])
    savi = norm01(ms[7])

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(rgb_like)
    axes[0].set_title("RGB-like")
    axes[0].axis("off")

    axes[1].imshow(false_color)
    axes[1].set_title("NIR-R-G")
    axes[1].axis("off")

    axes[2].imshow(ndvi)
    axes[2].set_title("NDVI")
    axes[2].axis("off")

    axes[3].imshow(gndvi)
    axes[3].set_title("GNDVI")
    axes[3].axis("off")

    axes[4].imshow(savi)
    axes[4].set_title("SAVI")
    axes[4].axis("off")

    fig.suptitle(f"id={sid} | true={true_label} | pred={pred_label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--mis_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mis_df = pd.read_csv(args.mis_csv)
    if "id" not in mis_df.columns:
        raise ValueError("mis_csv must contain column: id")
    if "true_label" not in mis_df.columns:
        raise ValueError("mis_csv must contain column: true_label")
    if "pred_label" not in mis_df.columns:
        raise ValueError("mis_csv must contain column: pred_label")

    ids = mis_df["id"].tolist()

    ds = DroughtDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        ids=ids,
        augment=False,
        normalize_method="percentile",
        target_size=(224, 224),
        modalities=["ms"],
    )

    print("num samples to export:", len(ds))

    for i in range(len(ds)):
        sid = mis_df.iloc[i]["id"]
        true_label = int(mis_df.iloc[i]["true_label"])
        pred_label = int(mis_df.iloc[i]["pred_label"])

        _, _, ms, _ = ds[i]
        ms = ms.numpy()

        out_name = f"id_{sid}_true_{true_label}_pred_{pred_label}.png"
        out_path = os.path.join(args.output_dir, out_name)

        save_sample_figure(ms, sid, true_label, pred_label, out_path)

    print("saved to:", args.output_dir)


if __name__ == "__main__":
    main()
