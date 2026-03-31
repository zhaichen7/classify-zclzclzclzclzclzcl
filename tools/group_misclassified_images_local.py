import re
import shutil
from pathlib import Path

SRC_DIR = Path("analysis_ms_confusion_bestsplit/vis_all_misclassified")
OUT_DIR = Path("analysis_ms_confusion_bestsplit/grouped_errors")

pattern = re.compile(r"id_(.+?)_true_(\d+)_pred_(\d+)\.png$", re.IGNORECASE)

OUT_DIR.mkdir(parents=True, exist_ok=True)

count_total = 0
count_matched = 0

if not SRC_DIR.exists():
    print("missing source dir:", SRC_DIR)
    raise SystemExit(1)

for file in SRC_DIR.iterdir():
    if not file.is_file():
        continue
    if file.suffix.lower() != ".png":
        continue

    count_total += 1
    m = pattern.match(file.name)
    if not m:
        print("unmatched filename:", file.name)
        continue

    sample_id, true_label, pred_label = m.groups()
    subdir = OUT_DIR / f"true_{true_label}_pred_{pred_label}"
    subdir.mkdir(parents=True, exist_ok=True)

    dst = subdir / file.name
    shutil.copy2(file, dst)
    count_matched += 1

print("done")
print("total files seen:", count_total)
print("matched and copied:", count_matched)
print("output dir:", OUT_DIR)
