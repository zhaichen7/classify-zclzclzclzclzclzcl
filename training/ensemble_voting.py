"""
ensemble_voting.py
集成学习：用投票融合现有的所有最好模型
预期: 55% → 58-62%
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

sys.path.append('.')
from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RestormerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8, dim=48, num_blocks=[4, 6],
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias'
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model(checkpoint_path):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RestormerClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def ensemble_predict(models, val_loader, model_names):
    """集成预测"""
    all_preds = []
    all_targets = []
    all_ensemble_preds = []
    
    print("\n进行集成预测...")
    with torch.no_grad():
        for _, _, ms, labels in val_loader:
            ms = ms.to(device)
            
            # 每个模型都预测
            batch_preds = []
            for model in models:
                output = model(ms)
                batch_preds.append(output.cpu().numpy())  # (B, 5)
            
            # 集成方式：概率平均后 argmax
            batch_preds = np.array(batch_preds)  # (num_models, B, 5)
            ensemble_output = batch_preds.mean(axis=0)  # (B, 5)
            ensemble_class = ensemble_output.argmax(axis=1)  # (B,)
            
            all_ensemble_preds.extend(ensemble_class)
            all_targets.extend(labels.numpy())
    
    return np.array(all_ensemble_preds), np.array(all_targets)

def main():
    print("="*70)
    print("🔗 集成学习 - 模型投票融合")
    print("="*70)
    
    # 加载数据
    print("\n📊 加载验证数据...")
    _, val_loader = build_dataloaders(
        csv_path='2025label_classic5.csv',
        data_root='dataset/',
        batch_size=16,
        num_workers=0,
        balanced=True,
        modalities=['ms']
    )
    
    print("✅ 数据加载完成")
    
    # 加载所有可用模型
    print("\n🧠 加载已有模型...")
    
    model_paths = [
        ('models_ms_opt_v1/drought_best.pth', '第1阶段 - 基础优化'),
        ('models_ms_opt_v2/drought_best.pth', '第2阶段 - 数据优化'),
        ('models_ms_opt_v3/drought_best.pth', '第3阶段 - ViT'),
    ]
    
    models = []
    model_names = []
    
    for path, name in model_paths:
        if os.path.exists(path):
            print(f"  ✅ 加载 {name}")
            model = load_model(path)
            models.append(model)
            model_names.append(name)
        else:
            print(f"  ⚠️  {path} 不存在，跳过")
    
    if len(models) < 2:
        print("\n❌ 至少需要 2 个模型，退出")
        return
    
    print(f"\n✅ 成功加载 {len(models)} 个模型")
    
    # 集成预测
    ensemble_preds, targets = ensemble_predict(models, val_loader, model_names)
    
    # 计算每个模型的性能
    print("\n" + "="*70)
    print("📊 各模型单独性能")
    print("="*70)
    
    results = []
    for model, name in zip(models, model_names):
        all_preds = []
        with torch.no_grad():
            for _, _, ms, labels in val_loader:
                ms = ms.to(device)
                output = model(ms)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        acc = accuracy_score(targets, all_preds)
        f1 = f1_score(targets, all_preds, average='weighted', zero_division=0)
        
        print(f"\n{name}:")
        print(f"  准确率: {acc*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
        results.append({'name': name, 'acc': acc, 'f1': f1})
    
    # 集成模型的性能
    ensemble_acc = accuracy_score(targets, ensemble_preds)
    ensemble_f1 = f1_score(targets, ensemble_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*70)
    print("🔗 集成模型性能")
    print("="*70)
    print(f"\n投票融合 ({len(models)} 个模型):")
    print(f"  准确率: {ensemble_acc*100:.2f}%")
    print(f"  F1分数: {ensemble_f1*100:.2f}%")
    
    # 对比总结
    print("\n" + "="*70)
    print("📈 性能对比")
    print("="*70)
    print(f"\n{'模型':<30} {'准确率':<15} {'F1分数':<15}")
    print("-"*60)
    
    for r in results:
        print(f"{r['name']:<30} {r['acc']*100:>6.2f}%{'':<7} {r['f1']*100:>6.2f}%")
    
    print(f"\n{'集成模型':<30} {ensemble_acc*100:>6.2f}%{'':<7} {ensemble_f1*100:>6.2f}%")
    
    # 计算提升
    best_single = max(results, key=lambda x: x['acc'])
    improvement = (ensemble_acc - best_single['acc']) * 100
    
    print(f"\n✅ 相对最优单模型提升: {improvement:+.2f}%")
    
    # 详细分类报告
    print("\n" + "="*70)
    print("📋 分类详细报告")
    print("="*70)
    print(classification_report(targets, ensemble_preds,
                               target_names=[f'Level {i}' for i in range(5)]))

if __name__ == '__main__':
    main()
