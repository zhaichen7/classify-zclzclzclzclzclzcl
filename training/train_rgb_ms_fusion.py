"""
train_rgb_ms_fusion.py
RGB + MS 注意力融合 - 五分类
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_drought import build_dataloaders

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        avg = self.fc(avg)
        max = self.fc(max)
        out = self.sigmoid(avg + max).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg, max], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

class MSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.MaxPool2d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.view(x.size(0), -1)

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        self.modality_weights = nn.Parameter(torch.ones(2))
    
    def forward(self, rgb_feat, ms_feat):
        features_stack = torch.stack([rgb_feat, ms_feat], dim=1)
        attended_features, attention_weights = self.attention(
            features_stack, features_stack, features_stack
        )
        features_stack = self.layer_norm1(features_stack + attended_features)
        ffn_output = self.ffn(features_stack)
        features_stack = self.layer_norm2(features_stack + ffn_output)
        weights = torch.softmax(self.modality_weights, dim=0)
        fused_feat = weights[0] * features_stack[:, 0] + weights[1] * features_stack[:, 1]
        return fused_feat, attention_weights

class RGBMSFusionNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.rgb_encoder = RGBEncoder()
        self.ms_encoder = MSEncoder()
        self.fusion = AttentionFusion(feature_dim=512, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, rgb, ms):
        rgb_feat = self.rgb_encoder(rgb)
        ms_feat = self.ms_encoder(ms)
        fused_feat, attn_weights = self.fusion(rgb_feat, ms_feat)
        logits = self.classifier(fused_feat)
        return logits, attn_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for rgb, _, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(rgb, ms)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False)
    for rgb, _, ms, labels in pbar:
        rgb = rgb.to(device)
        ms = ms.to(device)
        labels = labels.to(device)
        
        logits, _ = model(rgb, ms)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()
    
    class_accs = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(5)]
    return total_loss / total, correct / total, class_accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_rgb_ms_fusion")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🚀 RGB + MS 注意力融合 - 五分类")
    print("="*80)
    
    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=True,
        balanced=True,
        modalities=['rgb', 'ms']
    )
    print(f"✅ 数据加载完成")
    
    print("\n🧠 创建模型...")
    model = RGBMSFusionNet(num_classes=5)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    print("\n⚖️  计算类别权重...")
    labels_list = []
    for _, _, _, labels in train_loader:
        labels_list.extend(labels.numpy())
    
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*80)
    print("🚀 开始训练...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 35
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        
        if epoch % 15 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: LR={lr:.2e}, train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
            class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
            print(f"         Per-class: {class_str}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
            }, os.path.join(args.save_dir, 'fusion_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ 融合训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对MS单模态 (57.5%) 提升: {(best_val_acc - 0.575)*100:+.2f}%")
    print(f"预期目标: 65-75%，当前: {best_val_acc*100:.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'fusion_best.pth')}")
    print("="*80)

if __name__ == '__main__':
    main()
