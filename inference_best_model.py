"""
最终推理脚本 - 使用MS单模态55-57%模型
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 加载最佳模型
checkpoint = torch.load('./models_ms_opt_v1_new/drought_best.pth')
print(f"✅ 已加载最佳MS模型")
print(f"   验证集准确率: {checkpoint['val_acc']*100:.2f}%")
print(f"   各类准确率: {[f'c{i}={checkpoint['class_accs'][i]*100:.1f}%' for i in range(5)]}")
