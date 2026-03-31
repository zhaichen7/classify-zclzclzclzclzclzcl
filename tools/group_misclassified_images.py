import os
import re
import shutil
from pathlib import Path

SRC_DIRS = ["1", "2", "3"]
OUT_DIR = Path("grouped_errors")

pattern = re.compile(r"id_(.+?)_true_(\d+)_pred_(\d+)\.png$", re.IGNORECASE)

OUT_DIR.mkdir(exist_ok=True)

count_total = 0
count_matched = 0

for src in SRC_DIRS:
    src_path = Path(src)
    if not src_path.exists():
        print(f"skip missing dir: {src}")
        continue

    for file in src_path.iterdir():
        if not file.is_file():
            continue
        count_total += 1

        m = pattern.match(file.name)
        if not m:
            print(f"unmatched filename: {file}")
            continue

        sample_id, true_label, pred_label = m.groups()
        subdir = OUT_DIR / f"true_{true_label}_pred_{pred_label}"
        subdir.mkdir(parents=True, exist_ok=True)

        dst = subdir / file.name
        shutil.copy2(file, dst)
        count_matched += 1

print(f"done. total files seen: {count_total}")
print(f"matched and copied: {count_matched}")
print(f"output dir: {OUT_DIR}")
