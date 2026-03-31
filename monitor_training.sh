#!/bin/bash

echo "📊 实时监控训练进度"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "=============== 训练进度监控 ($(date '+%H:%M:%S')) ==============="
    echo ""
    
    # 获取最新的日志文件
    log_file=$(ls -1t logs/train_ms_opt_v1.log 2>/dev/null | head -1)
    
    if [ -f "$log_file" ]; then
        echo "📄 日志文件: $log_file"
        echo ""
        
        # 显示最后20行
        tail -30 "$log_file" | grep -E "Epoch|Per-class|保存|完成" | tail -10
        
        echo ""
        echo "📈 进度统计:"
        completed=$(tail -100 "$log_file" | grep "Epoch" | wc -l)
        echo "   已完成 Epoch 数: $completed"
        
        # 获取最佳准确率
        best_acc=$(tail -100 "$log_file" | grep "val_acc" | grep -oP 'val_acc=\K[0-9.]+' | sort -rn | head -1)
        if [ ! -z "$best_acc" ]; then
            echo "   当前最佳准确率: ${best_acc}%"
        fi
        
    else
        echo "⚠️  日志文件未找到，等待训练启动..."
    fi
    
    echo ""
    echo "下次更新: 5秒后"
    sleep 5
done
