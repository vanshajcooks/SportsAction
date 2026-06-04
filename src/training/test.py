# src/training/test.py
"""
Evaluation script for testing trained models.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_prep.dataset import SportsDataset
from src.models.video_classifier import VideoClassifier
from src.training.metrics import MetricsTracker

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"


def evaluate(checkpoint_path, backbone, temporal, num_frames=16, batch_size=4, device=None):
    """Evaluate model on test set."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    # Load dataset
    test_ds = SportsDataset(str(FRAMES_DIR), split="test", num_frames=num_frames, backbone=backbone)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(test_ds.classes)
    print(f"Classes: {test_ds.classes}")

    # Load model
    model = VideoClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        temporal=temporal,
        pretrained_backbone=False
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}\n")

    # Evaluate
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            preds = torch.argmax(logits, dim=1)
            metrics.update(preds, labels)

    # Print results
    metric_dict = metrics.get_metrics()
    print(f"Test Accuracy: {metric_dict.get('accuracy', 0.0)*100:.2f}%")
    print(f"Test F1-Score: {metric_dict.get('f1', 0.0):.4f}")
    print(f"Test Precision: {metric_dict.get('precision', 0.0):.4f}")
    print(f"Test Recall: {metric_dict.get('recall', 0.0):.4f}\n")

    # Confusion matrix and classification report
    all_preds = metrics.all_preds
    all_targets = metrics.all_targets
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=test_ds.classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained video action recognition model")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "densenet121", "inception_v3"],
                        help="Spatial backbone network")
    parser.add_argument("--temporal", type=str, default="bilstm",
                        choices=["none", "bilstm", "lstm", "gru", "tcn"],
                        help="Temporal modeling module")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Path to experiment dir (overrides backbone+temporal)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per clip")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    
    args = parser.parse_args()

    if args.exp_dir:
        ckpt = Path(args.exp_dir) / "best_model.pth"
    else:
        ckpt = PROJECT_ROOT / "experiments" / f"{args.backbone.lower()}_{args.temporal.lower()}" / "best_model.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    evaluate(
        ckpt,
        backbone=args.backbone,
        temporal=args.temporal,
        num_frames=args.num_frames,
        batch_size=args.batch_size
    )
