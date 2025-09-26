import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import SportsDataset
from src.models import VideoClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"

def evaluate(checkpoint_path, backbone, temporal, num_frames=16, batch_size=4, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Using device:", device)

    # Pass backbone to dataset for proper resizing
    test_ds = SportsDataset(str(DATA_ROOT), split="test", num_frames=num_frames, backbone=backbone)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(test_ds.classes)
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
    print(f"Loaded checkpoint: {checkpoint_path}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(int(p == l) for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nTest Accuracy: {acc*100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "densenet121", "inception_v3"])
    parser.add_argument("--temporal", type=str, default="bilstm", choices=["none", "bilstm", "tcn"])
    parser.add_argument("--exp_dir", type=str, default=None, help="optional: path to experiment dir (overrides backbone+temporal)")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if args.exp_dir:
        ckpt = Path(args.exp_dir) / "best_model.pth"
    else:
        ckpt = PROJECT_ROOT / "experiments" / f"{args.backbone.lower()}_{args.temporal.lower()}" / "best_model.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first or pass --exp_dir.")

    evaluate(ckpt, backbone=args.backbone, temporal=args.temporal, num_frames=args.num_frames, batch_size=args.batch_size)
