# train.py
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler
from src.dataset import SportsDataset
from src.models import VideoClassifier
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
FAILED_LOG = PROJECT_ROOT / "failed_files.log"

# ---------------- Collate function ----------------
def collate_skip_none(batch):
    """Skip any None items in a batch (unreadable videos)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ---------------- Validation ----------------
def validate(model, loader, device, criterion):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            clips, labels = batch
            clips, labels = clips.to(device), labels.to(device)
            with autocast("cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total

# ---------------- Training ----------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Device:", device)

    # Clear previous failed log
    if FAILED_LOG.exists():
        FAILED_LOG.unlink()

    # Datasets
    train_ds = SportsDataset(str(DATA_ROOT), split="train", num_frames=args.num_frames, backbone=args.backbone)
    val_ds   = SportsDataset(str(DATA_ROOT), split="val", num_frames=args.num_frames, backbone=args.backbone)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_skip_none)

    num_classes = len(train_ds.classes)
    print(f"Found {len(train_ds)} train samples, {len(val_ds)} val samples, {num_classes} classes.")

    # Experiment directory
    exp_name = f"{args.backbone.lower()}_{args.temporal.lower()}"
    EXP_DIR = PROJECT_ROOT / "experiments" / exp_name
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print("Experiment dir:", EXP_DIR)

    # Model
    model = VideoClassifier(
        backbone_name=args.backbone,
        num_classes=num_classes,
        temporal=args.temporal,
        pretrained_backbone=args.pretrained,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=not args.unidirectional
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            if batch is None:
                continue
            clips, labels = batch
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward
            with autocast("cuda"):
                logits = model(clips)
                loss = criterion(logits, labels)

            # Backward pass with scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            epoch_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

        scheduler.step()

        train_loss = epoch_loss / epoch_total if epoch_total else 0.0
        train_acc = epoch_correct / epoch_total if epoch_total else 0.0
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  time: {elapsed:.1f}s  "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}  "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = EXP_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args)
            }, ckpt_path)
            print(f"✅ Saved best model to {ckpt_path} (val_acc: {val_acc:.4f})")

    print("Training finished. Best val acc:", best_val_acc)
    if FAILED_LOG.exists():
        print(f"⚠️ Some files failed to load. Check {FAILED_LOG}.")

# ---------------- Argument parser ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "densenet121", "inception_v3"])
    parser.add_argument("--temporal", type=str, default="bilstm",
                        choices=["none", "bilstm", "tcn"])
    parser.add_argument("--num_frames", type=int, default=16, help="frames per clip")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_step", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="use pretrained backbone weights")
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--unidirectional", action="store_true", help="use unidirectional LSTM")
    parser.add_argument("--no_cuda", action="store_true", help="disable CUDA even if available")
    args = parser.parse_args()
    train(args)
