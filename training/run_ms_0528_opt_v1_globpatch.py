import os
import sys
import glob
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import datasets.dataset_drought as dd

orig_get_file_paths = dd.get_file_paths

def patched_get_file_paths(sample_id, data_root):
    """
    先沿用原来的路径逻辑把日期从 0519 替换成当前 data_root 的日期，
    然后不再信任旧的文件名模板，而是在正确目录里用 *_{id}.hdr 直接匹配真实文件。
    """
    paths = orig_get_file_paths(sample_id, data_root)
    date_tag = os.path.basename(os.path.normpath(data_root))   # 0528
    sid = str(sample_id)

    fixed = {}
    for k, v in paths.items():
        nv = v
        nv = nv.replace("_0519_", f"_{date_tag}_")
        nv = nv.replace("/0519_", f"/{date_tag}_")
        nv = nv.replace("0519_", f"{date_tag}_")

        folder = os.path.dirname(nv)
        matches = sorted(glob.glob(os.path.join(folder, f"*_{sid}.hdr")))

        if not matches:
            raise FileNotFoundError(
                f"[globpatch] no matched hdr for key={k}, sample_id={sid}, folder={folder}"
            )

        fixed[k] = matches[0]

    return fixed

dd.get_file_paths = patched_get_file_paths

# 直接运行你现有的 0528 训练脚本
target = os.path.join(ROOT, "training", "train_ms_0528_optimized_v1.py")
runpy.run_path(target, run_name="__main__")
