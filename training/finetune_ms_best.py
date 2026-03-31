"""
finetune_ms_best.py
基于57.5%最佳模型做fine-tune
策略: 冻结前面层，只微调分类头和最后几层
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_drought import build_dataloaders

class RestormerEncoder(nn.Module):
    """与之前57.5%模型相同的编码器"""
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
    def __init__(self, encoder_dim=192):
        super().__init__()
        self.encoder = RestormerEncoder(
            inp_channels=8, dim=48, num_blocks=[4, 6],
            heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias'
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

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

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, leave=False)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(ms)
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
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    pbar = tqdm(val_loader, leave=False)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        logits = model(ms)
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)  # 更小的学习率
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_finetuned")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrain_path", default="./models_ms_opt_v1_new/drought_best.pth")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🚀 MS单模态 - Fine-tune 57.5%最佳模型")
    print("="*80)
    
    print("\n📊 加载数据...")
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.2,
        random_state=42,
        augment_train=False,
        balanced=False,
        modalities=['ms']
    )
    print(f"✅ 数据加载完成")
    
    print("\n🧠 创建模型...")
    model = MSClassifier(encoder_dim=192)
    
    # 加载预训练权重
    print(f"\n📥 加载预训练模型: {args.pretrain_path}")
    checkpoint = torch.load(args.pretrain_path, map_location=device)
    
    # 尝试加载权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 预训练模型已加载 (Epoch {checkpoint['epoch']}, Acc {checkpoint['val_acc']*100:.2f}%)")
    except Exception as e:
        print(f"⚠️  权重加载有差异，继续训练: {e}")
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    # 冻结编码器，只训练分类头
    print("\n❄️  冻结编码器，只微调分类头...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 可训练参数: {trainable_params:,}")
    
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
    
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\n" + "="*80)
    print("🚀 开始微调...")
    print("="*80 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, class_accs = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
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
            }, os.path.join(args.save_dir, 'ms_finetuned.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ 微调完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"相对基础 (57.5%) 变化: {(best_val_acc - 0.575)*100:+.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'ms_finetuned.pth')}")
    print("="*80)

if __name__ == '__main__':
    main()
