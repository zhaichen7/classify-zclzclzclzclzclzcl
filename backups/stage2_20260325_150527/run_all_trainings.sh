#!/bin/bash
# run_all_trainings.sh
# 一键启动所有训练任务（后台并行）

set -e

echo "=========================================="
echo "干旱分级 - 完整训练管道"
echo "=========================================="
echo "当前目录: $(pwd)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /home/zcl/addfuse1

# 创建日志目录
mkdir -p logs

echo "[1/6] 启动 RGB 单模态训练..."
nohup python training/train_rgb_single.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_rgb_single \
  > logs/rgb_single.log 2>&1 &
echo "✅ RGB 训练已启动 (PID: $!)"

echo "[2/6] 启动 TIR 单模态训练..."
nohup python training/train_tir_single.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_tir_single \
  > logs/tir_single.log 2>&1 &
echo "✅ TIR 训练已启动 (PID: $!)"

echo "[3/6] 启动 MS 单模态训练..."
nohup python training/train_ms_single.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_ms_single \
  > logs/ms_single.log 2>&1 &
echo "✅ MS 训练已启动 (PID: $!)"

echo "[4/6] 启动 RGB+TIR 多模态训练..."
nohup python training/train_rgb_tir_multimodal.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_rgb_tir \
  > logs/rgb_tir.log 2>&1 &
echo "✅ RGB+TIR 训练已启动 (PID: $!)"

echo "[5/6] 启动 RGB+MS 多模态训练..."
nohup python training/train_rgb_ms_multimodal.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_rgb_ms \
  > logs/rgb_ms.log 2>&1 &
echo "✅ RGB+MS 训练已启动 (PID: $!)"

echo "[6/6] 启动 TIR+MS 多模态训练..."
nohup python training/train_tir_ms_multimodal.py \
  --csv_path 2025label_classic5.csv \
  --data_root dataset/ \
  --epochs 40 \
  --batch_size 4 \
  --save_dir ./models_tir_ms \
  > logs/tir_ms.log 2>&1 &
echo "✅ TIR+MS 训练已启动 (PID: $!)"

echo ""
echo "=========================================="
echo "✅ 所有训练任务已后台启动！"
echo "=========================================="
echo ""
echo "📋 查看训练进度:"
echo "  tail -f logs/rgb_single.log    # RGB 进度"
echo "  tail -f logs/tir_single.log    # TIR 进度"
echo "  tail -f logs/ms_single.log     # MS 进度"
echo "  tail -f logs/rgb_tir.log       # RGB+TIR 进度"
echo "  tail -f logs/rgb_ms.log        # RGB+MS 进度"
echo "  tail -f logs/tir_ms.log        # TIR+MS 进度"
echo ""
echo "📊 查看所有任务状态:"
echo "  jobs -l                        # 后台任务列表"
echo "  ps aux | grep python           # 所有 Python 进程"
echo ""
echo "📁 模型保存位置:"
echo "  ./models_rgb_single/"
echo "  ./models_tir_single/"
echo "  ./models_ms_single/"
echo "  ./models_rgb_tir/"
echo "  ./models_rgb_ms/"
echo "  ./models_tir_ms/"
echo ""
echo "⏱️  预计耗时: 每个训练 30-40 分钟"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

