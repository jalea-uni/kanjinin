import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ETLDataset
from model import get_model
import os

# Automatic device selection (supports MPS on Apple Silicon)
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Use mixed precision if available
USE_AMP = DEVICE.type in ['cuda', 'mps']


def main(args):
    # Load label map to determine num_classes
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    etl_dirs = args.etl_dirs.split(',')
    dataset = ETLDataset(args.data_path, etl_dirs, args.label_map, transform)

    # Split dataset
    n = len(dataset)
    val_size = test_size = int(n * args.val_split)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    num_workers = max(1, os.cpu_count() - 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Model, loss, optimizer
    model = get_model(num_classes, pretrained=args.pretrained).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Device: {DEVICE.type}]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/((loop.n+1)))
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                else:
                    outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = correct / total
        print(f"Val Acc: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print("New best model saved.")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Root folder containing ETL*/ subdirs')
    parser.add_argument('--etl-dirs', default="ETL1,ETL2,ETL3,ETL4,ETL5,ETL6,ETL7,ETL8G,ETL9G",
                        help='Comma-separated ETL directory names')
    parser.add_argument('--label-map', default='label_map.json', help='Path to label_map.json')
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=20, help='Reduce epochs to shorten training time')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--save-path', default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    main(args)