"""
ensemble_voting_fixed.py
集成学习：用投票融合 Restormer 模型
只用第1、2阶段（都是 Restormer）
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
    """加载 Restormer 模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RestormerClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def main():
    print("="*70)
    print("🔗 集成学习 - Restormer 模型投票融合")
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
    
    # 加载 Restormer 模型
    print("\n🧠 加载 Restormer 模型...")
    
    model_paths = [
        ('models_ms_opt_v1/drought_best.pth', '第1阶段 - 基础优化 (Focal Loss)'),
        ('models_ms_opt_v2/drought_best.pth', '第2阶段 - 数据优化 (更强正则)'),
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
            print(f"  ⚠️  {path} 不存在")
    
    if len(models) < 2:
        print("\n❌ 需要至少 2 个模型")
        return
    
    print(f"\n✅ 成功加载 {len(models)} 个 Restormer 模型")
    
    # 获取所有数据
    print("\n📊 进行推理...")
    all_targets = []
    all_model_preds = [[] for _ in range(len(models))]
    
    with torch.no_grad():
        for _, _, ms, labels in val_loader:
            ms = ms.to(device)
            all_targets.extend(labels.numpy())
            
            for i, model in enumerate(models):
                output = model(ms)
                preds = output.argmax(dim=1).cpu().numpy()
                all_model_preds[i].extend(preds)
    
    all_targets = np.array(all_targets)
    all_model_preds = [np.array(p) for p in all_model_preds]
    
    # 计算单个模型的性能
    print("\n" + "="*70)
    print("📊 各模型单独性能")
    print("="*70)
    
    results = []
    for i, (model_name, preds) in enumerate(zip(model_names, all_model_preds)):
        acc = accuracy_score(all_targets, preds)
        f1 = f1_score(all_targets, preds, average='weighted', zero_division=0)
        
        print(f"\n{model_name}:")
        print(f"  准确率: {acc*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
        
        results.append({'name': model_name, 'acc': acc, 'f1': f1, 'preds': preds})
    
    # 集成方法 1：投票 (多数投票)
    print("\n" + "="*70)
    print("🔗 集成方法 - 多数投票")
    print("="*70)
    
    # 对每个样本，三个模型投票
    stacked_preds = np.array(all_model_preds)  # (num_models, num_samples)
    
    # 方法 1: 简单投票
    voted_preds = []
    for j in range(stacked_preds.shape[1]):
        # 这个样本的所有模型预测
        votes = stacked_preds[:, j]
        # 选择最多的投票
        from collections import Counter
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common(1)[0][0]
        voted_preds.append(most_common)
    
    voted_preds = np.array(voted_preds)
    
    vote_acc = accuracy_score(all_targets, voted_preds)
    vote_f1 = f1_score(all_targets, voted_preds, average='weighted', zero_division=0)
    
    print(f"\n多数投票 (2 个模型):")
    print(f"  准确率: {vote_acc*100:.2f}%")
    print(f"  F1分数: {vote_f1*100:.2f}%")
    
    # 对比总结
    print("\n" + "="*70)
    print("📈 性能对比")
    print("="*70)
    print(f"\n{'模型':<40} {'准确率':<15} {'F1分数':<15}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<40} {r['acc']*100:>6.2f}%{'':<7} {r['f1']*100:>6.2f}%")
    
    print(f"\n{'集成模型 (投票)':<40} {vote_acc*100:>6.2f}%{'':<7} {vote_f1*100:>6.2f}%")
    
    # 计算提升
    best_single = max(results, key=lambda x: x['acc'])
    improvement = (vote_acc - best_single['acc']) * 100
    
    if improvement > 0:
        print(f"\n✅ 相对最优单模型提升: {improvement:+.2f}%")
    else:
        print(f"\n❌ 相对最优单模型下降: {improvement:+.2f}%")
    
    # 详细分类报告
    print("\n" + "="*70)
    print("📋 集成模型分类详细报告")
    print("="*70)
    print(classification_report(all_targets, voted_preds,
                               target_names=[f'Level {i}' for i in range(5)]))
    
    # 保存结果
    print("\n" + "="*70)
    print("💾 实验结论")
    print("="*70)
    print(f"\n当前最佳模型: {best_single['name']}")
    print(f"  单独准确率: {best_single['acc']*100:.2f}%")
    print(f"\n集成模型准确率: {vote_acc*100:.2f}%")
    
    if vote_acc > best_single['acc']:
        print(f"✅ 集成有效！提升了 {improvement:.2f}%")
        print(f"✅ 下一步: 继续尝试融合或其他方法")
    else:
        print(f"❌ 集成没有帮助")
        print(f"❌ 需要尝试其他方向（融合、超参优化等）")

if __name__ == '__main__':
    main()
