"""
评估K折交叉验证的结果
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys

sys.path.append('.')
from training.train_kfold_final_v6 import TrimodalFusionNet

# 这里填入从日志中看到的结果
fold_results = [
    {"fold": 1, "acc": 0.4706, "f1": 0.4714},
    {"fold": 2, "acc": 0.4471, "f1": 0.4222},
    {"fold": 3, "acc": 0.4706, "f1": 0.4508},
    {"fold": 4, "acc": 0.4286, "f1": 0.4265},
    {"fold": 5, "acc": 0.0, "f1": 0.0},  # 需要找到
]

print("="*80)
print("🎯 K折交叉验证最终结果")
print("="*80)

accs = []
f1s = []

for r in fold_results:
    if r['acc'] > 0:
        print(f"Fold {r['fold']}: Acc={r['acc']*100:.2f}%, F1={r['f1']*100:.2f}%")
        accs.append(r['acc'])
        f1s.append(r['f1'])

if accs:
    print(f"\n📊 平均性能:")
    print(f"  准确率: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  F1分数: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")

print("="*80)
