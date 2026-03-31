"""
train_ms_final_push.py
最后冲刺 - 基于V1最好的57.5%模型做fine-tune
策略:
  1. 加载V1最佳权重
  2. 解冻全部参数
  3. 用更小的学习率继续训练 (5e-5)
  4. 强化Focal Loss (gamma=2.3)
  5. 更强的Label Smoothing (0.15)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net_drought_rgb import RestormerEncoder
from datasets.dataset_drought import build_dataloaders

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
    for batch_idx, (rgb, tir, ms, labels) in enumerate(pbar):
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
    for batch_idx, (rgb, tir, ms, labels) in enumerate(pbar):
        ms = ms.to(device)
        labels = labels.to(device)
        
        outputs = model(ms)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(5):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += (preds[mask] == labels[mask]).sum().item()
    
    class_accs = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(5)]
    return total_loss / total, correct / total, class_accs

def main():
    parser = argparse.ArgumentParser(description='MS单模态最后冲刺')
    parser.add_argument('--csv_path', default='2025label_classic5.csv')
    parser.add_argument('--data_root', default='dataset/')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=2.3)
    parser.add_argument('--label_smoothing', type=float, default=0.15)
    parser.add_argument('--save_dir', default='./models_ms_final_push')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--pretrain_path', default='./models_ms_opt_v1_new/drought_best.pth')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("="*80)
    print("🚀 MS单模态 最后冲刺 - Fine-tune 57.5%最佳模型")
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
    model = RestormerEncoder(
        inp_channels=8, dim=48, num_blocks=[4, 6],
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias'
    ).to(device)
    
    classifier = nn.Sequential(
        nn.Linear(48, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 5)
    ).to(device)
    
    class FullModel(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = classifier
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = FullModel(model, classifier)
    
    # 加载预训练权重
    print(f"\n📥 加载预训练权重: {args.pretrain_path}")
    try:
        ckpt = torch.load(args.pretrain_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ 已加载 (Epoch {ckpt['epoch']}, Acc {ckpt['val_acc']*100:.2f}%)")
    except Exception as e:
        print(f"⚠️  加载失败: {e}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 总参数: {total_params:,}")
    
    print("\n⚖️  计算类别权重...")
    labels_list = []
    for _, _, _, labels in train_loader:
        labels_list.extend(labels.numpy())
    
    from collections import Counter
    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = torch.tensor(
        [total_samples / (5 * label_counts.get(i, 1)) for i in range(5)],
        dtype=torch.float, device=device
    )
    class_weights = class_weights / class_weights.sum() * 5
    
    focal_loss = FocalLoss(alpha=class_weights, gamma=args.gamma)
    label_smooth_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)
    
    def criterion(outputs, targets):
        return 0.6 * focal_loss(outputs, targets) + 0.4 * label_smooth_loss(outputs, targets)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    
    print("\n" + "="*80)
    print("🚀 开始Fine-tune训练...")
    print(f"超参数: LR={args.lr}, Gamma={args.gamma}, Label Smoothing={args.label_smoothing}")
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
        
        if epoch % 8 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: LR={lr:.2e}, train_acc={train_acc*100:.2f}%, val_acc={val_acc*100:.2f}%")
            class_str = " | ".join([f"c{i}={class_accs[i]*100:.1f}%" for i in range(5)])
            print(f"         {class_str}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_accs': class_accs,
            }, os.path.join(args.save_dir, 'drought_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️  早停 (Epoch {epoch})")
                break
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳准确率: {best_val_acc*100:.2f}%")
    print(f"相对基础 (57.5%) 变化: {(best_val_acc - 0.575)*100:+.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()
