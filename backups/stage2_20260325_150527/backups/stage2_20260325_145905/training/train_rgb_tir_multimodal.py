"""
RGB + TIR 双模态干旱分级训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.dataset_drought import build_dataloaders
from models.net_drought_rgb import DroughtClassifierRGB
import argparse
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for rgb, tir, _, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        labels = labels.to(device)
        
        # 将 RGB 和 TIR 拼接（都是3通道，总共6通道）
        combined = torch.cat([rgb, tir], dim=1)  # (B, 6, H, W)
        
        optimizer.zero_grad()
        outputs = model(combined)  # 模型接收6通道输入
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for rgb, tir, _, labels in pbar:
        rgb = rgb.to(device)
        tir = tir.to(device)
        labels = labels.to(device)
        
        combined = torch.cat([rgb, tir], dim=1)
        outputs = model(combined)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description="RGB+TIR 双模态干旱分级训练")
    parser.add_argument("--csv_path", type=str, default="2025label_classic5.csv")
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./models_rgb_tir")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*60)
    print("RGB+TIR 双模态干旱分级训练")
    print("="*60)
    
    # 加载数据（只需要 RGB 和 TIR）
    train_loader, val_loader = build_dataloaders(
        csv_path=args.csv_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        balanced=True,
        modalities=['rgb', 'tir'],  # 只加载 RGB 和 TIR
    )
    
    # 创建模型（输入通道：6 = 3(RGB) + 3(TIR)）
    model = DroughtClassifierRGB(dim=48, num_blocks=[4, 6], heads=[1, 2, 4, 8])
    # 修改第一层以接收6通道输入
    old_proj = model.encoder_rgb.patch_embed
    model.encoder_rgb.patch_embed = nn.Conv2d(6, 48, kernel_size=3, stride=1, padding=1, bias=False)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'drought_best.pth'))
    
    print(f"\n✅ RGB+TIR 模型训练完成! 最佳准确率: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    main()
