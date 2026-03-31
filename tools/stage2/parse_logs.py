#!/usr/bin/env python3
import re
import sys
import glob
from pathlib import Path

def norm_acc(x: str) -> float:
    v = float(x)
    return v * 100.0 if v <= 1.0 else v

def parse_one(path: str):
    best = -1.0
    best_epoch = None
    last_epoch = None
    found = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            em = re.search(r"Epoch\s+(\d+)", line, flags=re.I)
            if em:
                last_epoch = int(em.group(1))

            for m in re.finditer(r"(?:val[_ ]?acc(?:uracy)?|best[_ ]?acc(?:uracy)?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line, flags=re.I):
                found += 1
                acc = norm_acc(m.group(1))
                epoch = None
                em2 = re.search(r"Epoch\s+(\d+)", line, flags=re.I)
                if em2:
                    epoch = int(em2.group(1))
                elif last_epoch is not None:
                    epoch = last_epoch
                if acc > best:
                    best = acc
                    best_epoch = epoch

    return {
        "file": path,
        "best_acc": best,
        "best_epoch": best_epoch,
        "found": found,
    }

def main():
    files = []
    for arg in sys.argv[1:]:
        if "*" in arg or "?" in arg or "[" in arg:
            files.extend(glob.glob(arg))
        else:
            files.append(arg)

    files = [f for f in files if Path(f).is_file()]
    if not files:
        print("No log files found.")
        return

    rows = [parse_one(f) for f in files]
    rows.sort(key=lambda x: x["best_acc"], reverse=True)

    print("=" * 96)
    print(f"{'log_file':50} {'best_acc(%)':>12} {'best_epoch':>12} {'status':>12}")
    print("=" * 96)
    for r in rows:
        status = "OK" if r["found"] > 0 and r["best_acc"] >= 0 else "NO_VAL_ACC"
        best_acc = f"{r['best_acc']:.2f}" if r["best_acc"] >= 0 else "-"
        best_epoch = str(r["best_epoch"]) if r["best_epoch"] is not None else "-"
        print(f"{Path(r['file']).name:50} {best_acc:>12} {best_epoch:>12} {status:>12}")
    print("=" * 96)

if __name__ == "__main__":
    main()
