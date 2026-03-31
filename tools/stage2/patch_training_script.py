#!/usr/bin/env python3
import re
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: patch_training_script.py SRC DST [TAG]")
    sys.exit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else "stage2"

text = src.read_text(encoding="utf-8", errors="ignore")
changes = []

def subn(pattern, repl, name, flags=0):
    global text
    new_text, n = re.subn(pattern, repl, text, flags=flags)
    if n > 0:
        changes.append(f"{name}:{n}")
        text = new_text

subn(r'(?m)^(\s*(?:num_)?epochs\s*=\s*)40(\s*(?:#.*)?$)', r'\g<1>100\2', "epochs_eq")
subn(r'(?m)^(\s*max_epochs\s*=\s*)40(\s*(?:#.*)?$)', r'\g<1>100\2', "max_epochs_eq")
subn(r'(?m)(["\'](?:epochs|num_epochs|max_epochs)["\']\s*:\s*)40', r'\g<1>100', "epochs_dict")
subn(r'(?m)^(\s*(?:early_stop_patience|patience)\s*=\s*)\d+(\s*(?:#.*)?$)', r'\g<1>15\2', "patience_eq")
subn(r'(?m)(["\'](?:early_stop_patience|patience)["\']\s*:\s*)\d+', r'\g<1>15', "patience_dict")
subn(r'(?m)(\b(?:lr|learning_rate)\s*=\s*)(?:1e-?3|0\.001|5e-?4|0\.0005)', r'\g<1>3e-4', "lr_eq")
subn(r'(?m)(["\'](?:lr|learning_rate)["\']\s*:\s*)(?:1e-?3|0\.001|5e-?4|0\.0005)', r'\g<1>3e-4', "lr_dict")
subn(r'(?m)(\bweight_decay\s*=\s*)0(?:\.0+)?', r'\g<1>1e-4', "wd_eq")
subn(r'(?m)(["\']weight_decay["\']\s*:\s*)0(?:\.0+)?', r'\g<1>1e-4', "wd_dict")

def add_label_smoothing(match):
    inner = match.group(1)
    if "label_smoothing" in inner:
        return match.group(0)
    inner = inner.strip()
    if inner:
        return f"nn.CrossEntropyLoss({inner}, label_smoothing=0.05)"
    return "nn.CrossEntropyLoss(label_smoothing=0.05)"

new_text, n = re.subn(r'nn\.CrossEntropyLoss\(([^\n()]*)\)', add_label_smoothing, text)
if n > 0:
    changes.append(f"label_smoothing:{n}")
    text = new_text

def add_weight_decay(match):
    whole = match.group(0)
    if "weight_decay" in whole:
        return whole
    return whole[:-1] + ", weight_decay=1e-4)"

new_text, n = re.subn(r'optim\.(?:Adam|AdamW|SGD)\([^\n)]*\)', add_weight_decay, text)
if n > 0:
    changes.append(f"optimizer_wd:{n}")
    text = new_text

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(text, encoding="utf-8")

print(f"[PATCH] src={src}")
print(f"[PATCH] dst={dst}")
print(f"[PATCH] tag={tag}")
print(f"[PATCH] changes={', '.join(changes) if changes else 'none'}")
