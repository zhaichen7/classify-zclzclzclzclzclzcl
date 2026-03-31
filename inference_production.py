"""
inference_production.py
生产级推理脚本 - MS单模态57.5%最佳模型
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.dataset_drought import build_dataloaders

class RestormerEncoder(nn.Module):
    """Restormer编码器"""
    def __init__(self, inp_channels=8, dim=48, num_blocks=[4, 6], 
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, 
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1)
        
        self.encoder_level1 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks[0])
        ])
        self.down1 = nn.MaxPool2d(2)
        
        self.encoder_level2 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(dim, dim*2, 3, padding=1),
                nn.BatchNorm2d(dim*2),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks[1])
        ])
        self.down2 = nn.MaxPool2d(2)
        
        self.encoder_level3 = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(dim*2, dim*4, 3, padding=1),
                nn.BatchNorm2d(dim*4),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks[1])
        ])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder_level1(x)
        x = self.down1(x)
        x = self.encoder_level2(x)
        x = self.down2(x)
        x = self.encoder_level3(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

class MSClassifier(nn.Module):
    """MS分类模型"""
    def __init__(self):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8, dim=48, num_blocks=[4, 6],
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias'
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

def inference(model, data_loader, device):
    """推理函数"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="推理中")
        for _, _, ms, labels in pbar:
            ms = ms.to(device)
            logits = model(ms)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: {}".format(device))
    
    print("\n" + "="*80)
    print("🚀 生产级推理 - MS单模态57.5%最佳模型")
    print("="*80)
    
    print("\n📥 加载模型...")
    model = MSClassifier().to(device)
    checkpoint = torch.load('./models_ms_opt_v1_new/drought_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✅ 模型已加载")
    print("   训练Epoch: {}".format(checkpoint['epoch']))
    print("   验证准确率: {:.2f}%".format(checkpoint['val_acc']*100))
    
    class_acc_str = " | ".join(["c{}={:.1f}%".format(i, checkpoint['class_accs'][i]*100) for i in range(5)])
    print("   各类准确率: {}".format(class_acc_str))
    
    print("\n📊 加载测试数据...")
    _, test_loader = build_dataloaders(
        csv_path='2025label_classic5.csv',
        data_root='dataset/',
        batch_size=8,
        num_workers=4,
        test_size=0.2,
        random_state=42,
        augment_train=False,
        balanced=False,
        modalities=['ms']
    )
    print("✅ 数据已加载")
    
    print("\n🔍 开始推理...")
    preds, probs, labels = inference(model, test_loader, device)
    
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    print("\n" + "="*80)
    print("📊 推理结果")
    print("="*80)
    print("总体准确率 (Accuracy):  {:.2f}%".format(acc*100))
    print("精确率 (Precision):     {:.2f}%".format(precision*100))
    print("召回率 (Recall):        {:.2f}%".format(recall*100))
    print("F1分数:                 {:.2f}%".format(f1*100))
    
    print("\n📈 各类别详细指标:")
    print(classification_report(labels, preds, target_names=["Level {}".format(i) for i in range(5)]))
    
    print("\n🔢 混淆矩阵:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    
    print("\n💾 保存推理结果...")
    results_data = {
        'True_Label': labels,
        'Pred_Label': preds,
        'Confidence': np.max(probs, axis=1),
    }
    for i in range(5):
        results_data['Prob_Class_{}'.format(i)] = probs[:, i]
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('inference_results.csv', index=False)
    print("✅ 结果已保存到: inference_results.csv")
    
    print("\n📋 样本统计:")
    for i in range(5):
        true_count = (labels == i).sum()
        pred_count = (preds == i).sum()
        print("  Level {}: 真实={}, 预测={}".format(i, true_count, pred_count))
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print("="*80)

if __name__ == '__main__':
    main()
