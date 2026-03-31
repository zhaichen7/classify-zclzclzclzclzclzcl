"""
evaluate_all_experiments.py
自动评估所有实验结果并生成对比表格
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def collect_results():
    """收集所有实验结果"""
    results = []
    
    model_dirs = sorted([d for d in Path(".").glob("models_*") if d.is_dir()])
    
    for model_dir in model_dirs:
        exp_name = model_dir.name.replace("models_", "")
        best_model = model_dir / "drought_best.pth"
        
        if best_model.exists():
            print(f"找到实验: {exp_name}")
            results.append({
                "实验名称": exp_name,
                "模型路径": str(best_model),
                "状态": "✅ 完成"
            })
        else:
            print(f"⚠️  实验 {exp_name} 无模型文件")
    
    return pd.DataFrame(results)

def generate_summary():
    """生成实验汇总报告"""
    df = collect_results()
    
    print("\n" + "="*70)
    print("实验汇总报告")
    print("="*70)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"��实验数: {len(df)}")
    print(f"成功完成: {(df['状态'] == '✅ 完成').sum()}")
    print("\n" + str(df.to_string(index=False)))
    
    # 保存为 CSV
    df.to_csv("experiment_summary.csv", index=False)
    print(f"\n✅ 汇总报告已保存到 experiment_summary.csv")
    
    return df

if __name__ == "__main__":
    generate_summary()
