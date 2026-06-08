"""
Main training engine for Sports Action Segmentation.
"""
import time
import json
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_prep.dataset import TennisActionDataset
from src.models.spatial_extractor import SpatialExtractor
from src.models.temporal_brain import TemporalBrain
from src.training.metrics import MetricsTracker

FAILED_LOG = PROJECT_ROOT / "failed_files.log"


class EarlyStopping:
    """Early stopping callback based on validation metric."""
    
    def __init__(self, patience=10, metric="f1", mode="max", min_delta=0.0):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training stops
            metric (str): Metric to monitor ("f1", "accuracy", "loss")
            mode (str): "max" if higher is better, "min" if lower is better
            min_delta (float): Minimum change to qualify as improvement
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, current_value, epoch):
        """
        Check if training should stop.
        Returns True if training should stop, False otherwise.
        """
        if self.mode == "max":
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
            
        if improved:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def load_config(config_path="config.yaml"):
    with open(PROJECT_ROOT / config_path, "r") as file:
        return yaml.safe_load(file)

def collate_skip_none(batch):
    """Skip any None items in a batch (unreadable videos)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def validate(spatial, temporal, loader, device, criterion, config):
    spatial.eval()
    temporal.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            
            clips, labels = batch
            clips, labels = clips.to(device), labels.to(device)
            
            with autocast(enabled=config["training"]["mixed_precision"]):
                features = spatial(clips)
                logits = temporal(features) # Shape: [Batch, Frames, Classes]
                
                # Expand clip labels to all frames: [Batch] -> [Batch, Frames]
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).repeat(1, logits.size(1))
                
                # Flatten for loss calculation
                num_classes = logits.size(-1)
                loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))
            
            preds = torch.argmax(logits, dim=-1)
            metrics.update(preds, labels, loss.item())
            
    return metrics.get_metrics()

def train():
    config = load_config()
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if FAILED_LOG.exists(): FAILED_LOG.unlink()

    # Datasets
    print("Loading datasets...")
    train_ds = TennisActionDataset("config.yaml", split="train")
    val_ds = TennisActionDataset("config.yaml", split="val")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"], 
        collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["evaluation"]["batch_size"], shuffle=False,
        num_workers=config["evaluation"]["num_workers"], pin_memory=config["training"]["pin_memory"], 
        collate_fn=collate_skip_none
    )

    num_classes = len(train_ds.classes)
    
    # Directories
    exp_name = f"{config['model']['backbone']['name']}_{config['model']['temporal']['type']}"
    EXP_DIR = PROJECT_ROOT / config["output"]["experiments_dir"] / exp_name
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {EXP_DIR}")

    # Initialize Dual Models
    spatial_model = SpatialExtractor("config.yaml").to(device)
    temporal_model = TemporalBrain("config.yaml", input_dim=spatial_model.feature_dim, num_classes=num_classes).to(device)
    
    # Combine parameters for optimizer
    params = list(spatial_model.parameters()) + list(temporal_model.parameters())
    print(f"Total trainable parameters: {sum(p.numel() for p in params if p.requires_grad):,}")

    # Loss with label smoothing (improved generalization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        params, 
        lr=float(config["training"]["optimizer"]["lr"]), 
        weight_decay=float(config["training"]["optimizer"]["weight_decay"])
    )
    
    scheduler_type = config["training"]["scheduler"]["type"]
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["num_epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler"]["step_size"], gamma=config["training"]["scheduler"]["gamma"])
        
    scaler = GradScaler(enabled=config["training"]["mixed_precision"])
    
    # Early stopping based on validation F1
    early_stopping = EarlyStopping(patience=15, metric="f1", mode="max", min_delta=0.001)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    training_log = []
    num_epochs = config["training"]["num_epochs"]

    print("\nStarting Training...")
    for epoch in range(1, num_epochs + 1):
        spatial_model.train()
        temporal_model.train()
        metrics = MetricsTracker()
        t0 = time.time()

        for batch in train_loader:
            if batch is None: continue
            
            clips, labels = batch
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=config["training"]["mixed_precision"]):
                features = spatial_model(clips)
                logits = temporal_model(features)
                
                # Expand clip labels to frame labels
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).repeat(1, logits.size(1))
                
                # Flatten [Batch, Frames, Classes] -> [Batch*Frames, Classes] for Loss
                loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=config["training"]["gradient_clip"])
            
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=-1)
            metrics.update(preds, labels, loss.item())

        scheduler.step()

        # Validation Phase
        train_metrics = metrics.get_metrics()
        val_metrics = validate(spatial_model, temporal_model, val_loader, device, criterion, config)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{num_epochs} | {elapsed:.1f}s | "
              f"Train Loss: {train_metrics['avg_loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['avg_loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f}")

        training_log.append({
            'epoch': epoch,
            'train_loss': float(train_metrics['avg_loss']),
            'train_acc': float(train_metrics['accuracy']),
            'val_loss': float(val_metrics['avg_loss']),
            'val_acc': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics['f1']),
            'elapsed_time': elapsed
        })

        # Save Best Model (based on F1 for better generalization)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_acc = val_metrics['accuracy']
            ckpt_path = EXP_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "spatial_state": spatial_model.state_dict(),
                "temporal_state": temporal_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_f1": best_val_f1,
            }, ckpt_path)
            print(f"✅ Best model saved (val_f1: {best_val_f1:.4f}, val_acc: {best_val_acc:.4f})")
        
        # Early stopping check
        if early_stopping(val_metrics['f1'], epoch):
            print(f"\n⏸️  Early stopping triggered at epoch {epoch}")
            print(f"Best F1 score: {best_val_f1:.4f} at epoch {early_stopping.best_epoch}")
            break

    # Save training log
    with open(EXP_DIR / "training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n✅ Training complete. Best val F1: {best_val_f1:.4f}, Best val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()