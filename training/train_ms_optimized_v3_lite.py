"""
train_ms_optimized_v3_lite.py - 轻量级 ViT
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_drought import build_dataloaders

# ============================================================================
# 轻量级 Vision Transformer
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=8, patch_size=16, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class LiteViT(nn.Module):
    def __init__(self, in_channels=8, patch_size=16, embed_dim=192, depth=6, num_heads=4, num_classes=5):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (224 // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, drop=0.1)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        
        return x

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# 训练函数
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.2f}%'})
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=True)
    for _, _, ms, labels in pbar:
        ms = ms.to(device)
        labels = labels.to(device)
        
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total*100:.2f}%'})
    
    return total_loss / total, correct / total

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="2025label_classic5.csv")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="./models_ms_opt_v3")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("🚀 第3阶段 - 轻量级 ViT 训练")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    # 加载数据
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
        modalities=['ms']
    )
    
    print(f"✅ 数据加载完成")
    
    # 创建模型
    print("\n🧠 创建 ViT 模型...")
    model = LiteViT(
        in_channels=8,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=4,
        num_classes=5
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数数: {total_params:,}")
    
    # 计算类别权重
    print("\n⚖️  计算类别权重...")
    from collections import Counter
    train_labels = []
    for _, _, _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    label_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = []
    for i in range(5):
        count = label_counts.get(i, 1)
        weight = total_samples / (5 * count)
        class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"✅ 类别权重: {[f'{w:.3f}' for w in class_weights.cpu().numpy()]}")
    
    # 损失函数
    print("\n🎯 创建损失函数...")
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.5 * focal_loss(outputs, targets) + 0.5 * label_smooth_loss(outputs, targets)
    
    # 优化器
    print("\n⚙️  创建优化器...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练循环
    print("\n" + "="*70)
    print("🚀 开始训练...")
    print("="*70 + "\n")
    
    best_val_acc = 0
    best_epoch = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"\n✓ Epoch {epoch:3d}/{args.epochs} | LR={lr:.2e} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'drought_best.pth'))
            
            print(f"           ✅ 保存最佳模型")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发")
                break
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ 第3阶段训练完成！")
    print("="*70)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳验证准确率: {best_val_acc*100:.2f}%")
    print(f"模型保存: {os.path.join(args.save_dir, 'drought_best.pth')}")
    print("="*70)

if __name__ == '__main__':
    main()
