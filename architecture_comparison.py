"""
architecture_comparison.py
测试不同架构组合的性能
- Restormer (基础)
- Restormer + DenseNet + HybridSN (之前的配置)
- 其他组合
"""
import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ARCH_EXPERIMENTS = [
    # 纯 Restormer
    {
        "name": "All_Restormer",
        "desc": "RGB=Restormer | TIR=Restormer | MS=Restormer",
        "rgb_arch": "restormer",
        "tir_arch": "restormer",
        "ms_arch": "restormer",
    },
    # 混合架构（之前表现较好的）
    {
        "name": "Mixed_RestormerDenseNetHybridSN",
        "desc": "RGB=Restormer | TIR=DenseNet | MS=HybridSN",
        "rgb_arch": "restormer",
        "tir_arch": "densenet",
        "ms_arch": "hybridsn",
    },
    # 其他尝试的组合
    {
        "name": "All_DenseNet",
        "desc": "RGB=DenseNet | TIR=DenseNet | MS=DenseNet",
        "rgb_arch": "densenet",
        "tir_arch": "densenet",
        "ms_arch": "densenet",
    },
    {
        "name": "All_HybridSN",
        "desc": "RGB=HybridSN | TIR=HybridSN | MS=HybridSN",
        "rgb_arch": "hybridsn",
        "tir_arch": "hybridsn",
        "ms_arch": "hybridsn",
    },
    {
        "name": "Restormer_DenseNet_Restormer",
        "desc": "RGB=Restormer | TIR=DenseNet | MS=Restormer",
        "rgb_arch": "restormer",
        "tir_arch": "densenet",
        "ms_arch": "restormer",
    },
]

def run_architecture_experiment(exp_config, csv_path, data_root):
    """运行架构组合实验"""
    exp_name = exp_config["name"]
    
    print(f"\n{'='*70}")
    print(f"架构实验: {exp_name}")
    print(f"描述: {exp_config['desc']}")
    print(f"{'='*70}")
    
    # 生成临时训练脚本
    script_content = f"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.net_drought import DroughtClassifier
from datasets.dataset_drought import build_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {{device}}")
    print(f"RGB arch: {exp_config['rgb_arch']}")
    print(f"TIR arch: {exp_config['tir_arch']}")
    print(f"MS arch: {exp_config['ms_arch']}")
    
    # 加载数据
    train_loader, val_loader = build_dataloaders(
        csv_path="{csv_path}",
        data_root="{data_root}",
        batch_size=4,
        num_workers=4,
        balanced=True,
    )
    
    # 创建模型
    model = DroughtClassifier(dim=48).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(1, 41):
        model.train()
        for rgb, tir, ms, labels in tqdm(train_loader, desc=f"Epoch {{epoch}}"):
            rgb, tir, ms, labels = rgb.to(device), tir.to(device), ms.to(device), labels.to(device)
            outputs = model(rgb, tir, ms)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for rgb, tir, ms, labels in val_loader:
                rgb, tir, ms, labels = rgb.to(device), tir.to(device), ms.to(device), labels.to(device)
                outputs = model(rgb, tir, ms)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {{epoch}}: val_acc = {{val_acc*100:.2f}}%")
    
    print(f"\\n最佳验证准确率: {{best_val_acc*100:.2f}}%")

if __name__ == "__main__":
    main()
"""
    
    # 执行脚本
    try:
        exec(script_content)
        return {"status": "success", "arch": exp_name}
    except Exception as e:
        print(f"❌ 架构 {exp_name} 失败: {e}")
        return {"status": "failed", "arch": exp_name}

def main():
    csv_path = "2025label_classic5.csv"
    data_root = "dataset/"
    
    print("\n" + "="*70)
    print("架构组合对比实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for i, exp_config in enumerate(ARCH_EXPERIMENTS, 1):
        print(f"\n[{i}/{len(ARCH_EXPERIMENTS)}]")
        result = run_architecture_experiment(exp_config, csv_path, data_root)
        results.append(result)
    
    print("\n" + "="*70)
    print("架构对比结果")
    print("="*70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with open("architecture_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
