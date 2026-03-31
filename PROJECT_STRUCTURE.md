# 干旱分级项目结构说明

## 📁 项目文件夹结构

```
addfuse/
├── models/              # 模型定义文件
│   ├── net_drought.py              # 核心干旱分类网络
│   └── net_drought_improved.py     # 改进版网络模型
├── datasets/            # 数据集处理文件
│   └── dataset_drought.py          # 干旱数据集加载器
├── training/            # 训练脚本
│   ├── train_binary_lite.py        # 二分类训练（效果很好）
│   ├── train_drought_fixed_v2.py   # 修复版多分类训练
│   └── train_drought_optimized_v4.py # 优化版多分类训练
├── utils/               # 工具和辅助函数
│   ├── advanced_augmentation.py    # 高级数据增强
│   ├── fusion_module.py            # 特征融合模块
│   ├── improved_loss.py            # 改进损失函数
│   ├── patch_dataset_augmentation.py # 补丁数据增强
│   └── preprocess_data.py          # 数据预处理脚本
├── evaluation/          # 评估和测试脚本
│   ├── evaluate_drought.py         # 模型评估
│   ├── plot_training_curves.py     # 训练曲线绘制
│   └── test_drought.py             # 模型测试
└── configs/             # 配置文件（预留）
```

## 🎯 各文件夹用途说明

### models/ - 模型定义
- **核心网络架构**：包含干旱分类的主要神经网络模型
- **模型改进**：包含优化后的网络结构

### datasets/ - 数据集处理
- **数据加载**：处理ENVI格式的遥感数据
- **预处理**：归一化、数据增强等预处理操作

### training/ - 训练脚本
- **二分类训练**：`train_binary_lite.py`（效果很好）
- **多分类训练**：修复版和优化版训练脚本
- **不同配置**：提供多种训练配置选择

### utils/ - 工具函数
- **数据增强**：高级数据增强技术
- **特征融合**：多模态特征融合模块
- **损失函数**：改进的损失函数实现
- **预处理**：数据预处理工具

### evaluation/ - 评估工具
- **模型评估**：性能评估和指标计算
- **可视化**：训练过程可视化
- **测试**：模型测试脚本

## 🚀 快速开始

### 二分类训练（推荐）
```bash
cd training/
python train_binary_lite.py
```

### 多分类训练
```bash
cd training/
python train_drought_fixed_v2.py
# 或
python train_drought_optimized_v4.py
```

### 数据预处理
```bash
cd utils/
python preprocess_data.py
```

### 模型评估
```bash
cd evaluation/
python evaluate_drought.py
```

## 📊 文件分类原则

1. **按功能分类**：相同功能的文件放在同一文件夹
2. **模块化设计**：每个文件夹职责明确
3. **易于维护**：结构清晰，便于扩展和维护
4. **标准化命名**：文件名清晰反映功能

## 💡 使用建议

- **二分类任务**：优先使用 `training/train_binary_lite.py`
- **多分类任务**：根据需求选择修复版或优化版
- **数据预处理**：使用 `utils/preprocess_data.py` 加速训练
- **模型评估**：使用 `evaluation/` 文件夹中的工具

---
*项目结构整理完成时间: 2025-03-11*