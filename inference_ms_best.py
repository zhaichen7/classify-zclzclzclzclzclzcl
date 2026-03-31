"""
inference_ms_best.py
用最好的MS模型 (57.5%) 做推理
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.dataset_drought import build_dataloaders

# 导入之前的模型
from training.train_ms_optimized_v1 import RestormerEncoder

class MSClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载最佳模型
print("🚀 加载最佳MS模型 (57.5%)...")
encoder = RestormerEncoder(
    inp_channels=8,
    dim=48,
    num_blocks=[4, 6],
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='WithBias'
)
model = MSClassifier(encoder)

checkpoint = torch.load('models_ms_opt_v1_new/drought_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ 模型已加载")
print(f"   最佳Epoch: {checkpoint['epoch']}")
print(f"   最佳准确率: {checkpoint['val_acc']*100:.2f}%")

# 加载测试数据
print("\n📊 加载测试数据...")
_, val_loader = build_dataloaders(
    csv_path='2025label_classic5.csv',
    data_root='dataset/',
    batch_size=4,
    num_workers=4,
    test_size=0.2,
    random_state=42,
    augment_train=False,
    balanced=False,
    modalities=['ms']
)

# 推理
print("\n🔍 开始推理...")
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for _, _, ms, labels in val_loader:
        ms = ms.to(device)
        logits = model(ms)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# 计算指标
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print("\n" + "="*80)
print("📊 推理结果 (MS单模态 57.5% 最佳模型)")
print("="*80)
print(f"准确率 (Accuracy):  {acc*100:.2f}%")
print(f"精确率 (Precision): {precision*100:.2f}%")
print(f"召回率 (Recall):    {recall*100:.2f}%")
print(f"F1分数:             {f1*100:.2f}%")
print("="*80)

# 类别分析
print("\n📈 各类别准确率:")
for i in range(5):
    mask = all_labels == i
    if mask.sum() > 0:
        class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        print(f"  类别 {i}: {class_acc*100:.2f}% ({mask.sum()} 样本)")
