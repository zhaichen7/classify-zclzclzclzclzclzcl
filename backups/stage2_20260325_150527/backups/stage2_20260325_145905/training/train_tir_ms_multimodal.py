import os, sys, argparse, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.net_drought import DroughtClassifier
from datasets.dataset_drought import build_dataloaders

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for _, tir, ms, labels in tqdm(loader, desc="Train", leave=False):
        tir, ms, labels = tir.to(device), ms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(torch.zeros_like(tir).to(device), tir, ms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for _, tir, ms, labels in tqdm(loader, desc="Val", leave=False):
        tir, ms, labels = tir.to(device), ms.to(device), labels.to(device)
        outputs = model(torch.zeros_like(tir).to(device), tir, ms)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default="2025label_classic5.csv")
parser.add_argument("--data_root", default="dataset/")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_dir", default="./models_tir_ms")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = build_dataloaders(
    csv_path=args.csv_path, data_root=args.data_root, batch_size=args.batch_size
)

model = DroughtClassifier(dim=48).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_val_acc = 0
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc}, 
                   os.path.join(args.save_dir, 'drought_best.pth'))

print(f"✅ TIR+MS 模型完成! 最佳准确率: {best_val_acc*100:.2f}%")
