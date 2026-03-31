"""
branch_ablation_study.py
单分支消融实验：逐个测试 RGB/TIR/MS 的贡献度
"""
import os
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 实验配置
EXPERIMENTS = [
    # 单分支实验（Restormer）
    {
        "name": "RGB_Only_Restormer",
        "desc": "仅 RGB 分支（Restormer）",
        "modalities": ["rgb"],
        "script": "training/train_rgb_single.py",
    },
    {
        "name": "TIR_Only_Restormer",
        "desc": "仅 TIR 分支（Restormer）",
        "modalities": ["tir"],
        "script": "training/train_tir_single.py",
    },
    {
        "name": "MS_Only_Restormer",
        "desc": "仅 MS 分支（Restormer）",
        "modalities": ["ms"],
        "script": "training/train_ms_single.py",
    },
    # 双分支实验 - 暂时用三分支模型跳过模态
    {
        "name": "RGB_TIR_Restormer",
        "desc": "RGB + TIR（不含 MS）",
        "modalities": ["rgb", "tir"],
        "script": "training/train_rgb_tir_multimodal.py",
    },
    {
        "name": "RGB_MS_Restormer",
        "desc": "RGB + MS（不含 TIR）",
        "modalities": ["rgb", "ms"],
        "script": "training/train_rgb_ms_multimodal.py",
    },
    {
        "name": "TIR_MS_Restormer",
        "desc": "TIR + MS（不含 RGB）",
        "modalities": ["tir", "ms"],
        "script": "training/train_tir_ms_multimodal.py",
    },
    # 三分支完整模型
    {
        "name": "RGB_TIR_MS_Restormer",
        "desc": "RGB + TIR + MS（完整）",
        "modalities": ["rgb", "tir", "ms"],
        "script": "training/train_drought_optimized_v4.py",
    },
]

def run_experiment(exp_config, csv_path, data_root, epochs=30, batch_size=4):
    """运行单个实验"""
    exp_name = exp_config["name"]
    script = exp_config["script"]
    
    print(f"\n{'='*70}")
    print(f"实验: {exp_name}")
    print(f"描述: {exp_config['desc']}")
    print(f"脚本: {script}")
    print(f"{'='*70}")
    
    # 检查脚本是否存在
    if not os.path.exists(script):
        print(f"⚠️  脚本不存在: {script}")
        return {"status": "missing", "exp": exp_name, "script": script}
    
    cmd = [
        "python", script,
        "--csv_path", csv_path,
        "--data_root", data_root,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--save_dir", f"./models_{exp_name}",
    ]
    
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            print(f"✅ 实验 {exp_name} 完成")
            return {"status": "success", "exp": exp_name}
        else:
            print(f"❌ 实验 {exp_name} 失败")
            error_msg = result.stderr[-500:] if result.stderr else "No error message"
            print(f"错误: {error_msg}")
            return {"status": "failed", "exp": exp_name, "error": error_msg}
    except subprocess.TimeoutExpired:
        print(f"❌ 实验 {exp_name} 超时")
        return {"status": "timeout", "exp": exp_name}
    except Exception as e:
        print(f"❌ 实验 {exp_name} 异常: {e}")
        return {"status": "error", "exp": exp_name, "error": str(e)}

def main():
    csv_path = "2025label_classic5.csv"
    data_root = "dataset/"
    
    print("\n" + "="*70)
    print("干旱分级单分支消融实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] 运行实验...")
        result = run_experiment(exp_config, csv_path, data_root, epochs=40, batch_size=4)
        results.append(result)
    
    # 汇总结果
    print("\n" + "="*70)
    print("实验汇总")
    print("="*70)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"成功: {success_count}/{len(EXPERIMENTS)}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印详细结果
    print("\n各实验状态:")
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"  {status_icon} {r['exp']}: {r['status']}")
    
    # 保存结果
    with open("ablation_study_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 结果已保存到 ablation_study_results.json")

if __name__ == "__main__":
    main()
