#!/bin/bash
# run_all_experiments.sh
# 一键运行所有消融实验和架构对比

set -e

echo "=================================================="
echo "干旱分级 - 完整实验管道"
echo "=================================================="

cd /home/zcl/addfuse1

# 1. 单分支消融实验
echo ""
echo "[1/3] 运行单分支消融实验..."
python branch_ablation_study.py | tee ablation_study.log

# 2. 架构对比实验  
echo ""
echo "[2/3] 运行架构对比实验..."
python architecture_comparison.py | tee architecture_comparison.log

# 3. 评估和汇总
echo ""
echo "[3/3] 生成实验汇总报告..."
python evaluate_all_experiments.py

echo ""
echo "=================================================="
echo "✅ 所有实验完成！"
echo "=================================================="
echo ""
echo "结果文件:"
echo "  - ablation_study_results.json"
echo "  - architecture_results.json"
echo "  - experiment_summary.csv"
echo ""

