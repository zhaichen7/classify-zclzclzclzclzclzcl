# 干旱分级五分类系统（CDDFuse 改造版）

基于 [CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)（CVPR 2023）图像融合框架改造的**干旱分级五分类**深度学习系统，使用三分支 Restormer 编码器融合 RGB、热红外和多光谱+植被指数数据。

---

## 目录结构

```
addfuse/
├── net_drought.py        # 三分支分类网络
├── dataset_drought.py    # ENVI 格式数据集加载器
├── train_drought.py      # 训练脚本
├── test_drought.py       # 测试/评估脚本
├── README_drought.md     # 本文档
└── models/               # 训练好的模型（自动创建）
    └── drought_best.pth
```

---

## 环境配置

### 1. 创建 conda 环境

```bash
conda create -n addfuse python=3.9
conda activate addfuse
```

### 2. 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn spectral matplotlib seaborn tqdm einops
```

或直接：

```bash
pip install -r requirements.txt
```

### requirements.txt 内容

```
torch>=1.12
torchvision>=0.13
numpy>=1.21
pandas>=1.4
scikit-learn>=1.1
spectral>=0.23
matplotlib>=3.5
seaborn>=0.12
tqdm>=4.64
einops>=0.4
```

---

## 数据准备

### 数据目录结构（服务器）

```
/home/zcl/addfuse/dataset/
  _0519_rgb_control/
    _0519_20m_kejianguang_1.dat
    _0519_20m_kejianguang_1.dat.enp
    ...
  _0519_rehongwai_control/
    _0519_rehongwai_20m_1.dat
    _0519_rehongwai_20m_1.dat.enp
    ...
  _0519_nir_control/
    _0519_duoguangpu_20m_840_1.dat
    _0519_duoguangpu_20m_840_1.dat.enp
    ...
  _0519_red_control/
    _0519_duoguangpu_20m_660_1.dat
    ...
  _0519_green_control/
    _0519_duoguangpu_20m_555_1.dat
    ...
  _0519_blue_control/
    _0519_duoguangpu_20m_450_1.dat
    ...
  _0519_rededge_control/
    _0519_duoguangpu_20m_720_1.dat
    ...
```

### 标签文件格式（CSV）

`/home/zcl/addfuse/2025label_classic5.csv` 需包含以下列：

| 列名  | 说明                 |
|-------|----------------------|
| ID    | 样本编号（从 1 开始） |
| label | 干旱等级（0-4）      |

---

## 网络结构

```
RGB (3,H,W)      ─→ Restormer_Encoder ─→ base_feature1 (64,H,W)
热红外 (3,H,W)    ─→ Restormer_Encoder ─→ base_feature2 (64,H,W)
多光谱+VI (8,H,W) ─→ Restormer_Encoder ─→ base_feature3 (64,H,W)
                              ↓
                 torch.cat([f1, f2, f3], dim=1)  → (192, H, W)
                              ↓
                    Global Average Pooling        → (192,)
                              ↓
                    FC(192 → 256) → ReLU → Dropout(0.3)
                              ↓
                    FC(256 → 5)
                              ↓
                    Softmax → 干旱等级 (0-4)
```

### 多光谱输入通道说明（8通道）

| 通道 | 内容     | 来源文件                    |
|------|----------|-----------------------------|
| 0    | NIR      | `_0519_duoguangpu_20m_840_*.dat` band 0 |
| 1    | Red      | `_0519_duoguangpu_20m_660_*.dat` band 0 |
| 2    | Blue     | `_0519_duoguangpu_20m_450_*.dat` band 0 |
| 3    | Green    | `_0519_duoguangpu_20m_555_*.dat` band 0 |
| 4    | RedEdge  | `_0519_duoguangpu_20m_720_*.dat` band 0 |
| 5    | NDVI     | (NIR - Red) / (NIR + Red)   |
| 6    | GNDVI    | (NIR - Green) / (NIR + Green) |
| 7    | SAVI     | (1+L)×(NIR-Red)/(NIR+Red+L), L=0.5 |

> **注意**：每个多光谱 `.dat` 文件的 band 0 为实测数据，band 1 为掩码（无效像素标记）。代码仅读取 band 0（`band_idx=0`）。

---

## 训练

```bash
python train_drought.py
```

### 默认超参数

| 参数          | 值     |
|---------------|--------|
| Batch size    | 8      |
| Epochs        | 100    |
| Optimizer     | Adam   |
| Learning rate | 1e-4   |
| LR scheduler  | StepLR (step=30, γ=0.5) |
| Loss          | CrossEntropyLoss |
| Train/Val 比例 | 80% / 20% |
| 随机种子      | 42     |

### 训练日志示例

```
Epoch 1/100  (lr=1.00e-04)
Training: 100%|██████████| 43/43 [00:30<00:00, loss: 1.2345, acc: 45.23%]
Validation: 100%|██████████| 11/11 [00:05<00:00]
Train Loss: 1.2345, Acc: 45.23%  |  Val Loss: 1.1234, Acc: 48.56%  [35.2s]
✅ Best model saved! Val Acc: 48.56%
```

最佳模型保存至 `./models/drought_best.pth`。

---

## 测试

```bash
python test_drought.py
```

### 输出结果

- **终端输出**：总体准确率、每类准确率、F1-score、分类报告
- **`./test_results/metrics.csv`**：数值指标汇总
- **`./test_results/confusion_matrix.png`**：混淆矩阵热力图

### 测试输出示例

```
=== Overall Accuracy: 72.94% ===

Per-class Accuracy:
  Level 0: 80.00%
  Level 1: 65.00%
  Level 2: 70.00%
  Level 3: 75.00%
  Level 4: 78.00%

Macro F1-score: 0.7123

Classification Report:
              precision    recall  f1-score   support
     Level 0       0.82      0.80      0.81        20
     Level 1       0.67      0.65      0.66        17
     ...
```

---

## 注意事项

1. **ENVI 格式**：每个 `.dat` 文件需要对应的 `.dat.enp` 头文件，缺少头文件会导致读取失败。
2. **多光谱 band 索引**：每个多光谱 `.dat` 文件包含两个 band：band 0 为实测数据，band 1 为掩码（无效像素标记）。代码仅读取 band 0（`band_idx=0`），不使用掩码 band。
3. **归一化**：默认使用百分位数归一化（2%-98%），可在 `dataset_drought.py` 中改为 `minmax`。
4. **数据增强**：训练时默认启用随机水平/垂直翻转，可在 `build_dataloaders` 中设置 `augment_train=False` 关闭。
5. **修改路径**：如数据路径不同，请修改 `train_drought.py` 和 `test_drought.py` 中的 `CSV_PATH` 和 `DATA_ROOT`。
