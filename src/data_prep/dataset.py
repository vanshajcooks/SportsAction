import os
import torch
import yaml
import glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_config(config_path):
    """Loads the global configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class TennisActionDataset(Dataset):
    def __init__(self, config_path, split="train"):
        """
        Args:
            config_path (str): Path to config.yaml
            split (str): "train", "val", or "test"
        """
        self.config = load_config(config_path)
        self.frames_dir = Path(self.config["data"]["frames_dir"]) / split
        self.num_frames = self.config["training"]["num_frames"]
        self.frame_size = self.config["data"]["frame_size"]
        
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {self.frames_dir}")

        # Map action subfolders directly to class IDs (e.g., 'backhand' -> 0, 'forehand' -> 1)
        self.classes = sorted([d.name for d in self.frames_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Gather all individual video frame directories and their corresponding class labels
        self.video_folders = []
        for cls_name in self.classes:
            cls_dir = self.frames_dir / cls_name
            for video_dir in cls_dir.iterdir():
                if video_dir.is_dir():
                    self.video_folders.append((video_dir, self.class_to_idx[cls_name]))

        # ImageNet normalization transforms required for pre-trained CNN backbones
        self.transform = transforms.Compose([
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_folders)

    def _sample_frames(self, frame_paths):
        """Uniformly samples or pads frames to ensure an exact temporal sequence length."""
        total_frames = len(frame_paths)
        if total_frames == 0:
            raise ValueError("Encountered an empty video folder during sampling.")

        # Temporal Padding: If video is too short, repeat the final frame
        if total_frames < self.num_frames:
            padding = [frame_paths[-1]] * (self.num_frames - total_frames)
            frame_paths.extend(padding)
            return frame_paths

        # Temporal Subsampling: Evenly space out frame selections across the window
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx):
        video_dir, label = self.video_folders[idx]
        
        # Ensure sequential frame order matching filename sort
        frame_paths = sorted(glob.glob(str(video_dir / "*.jpg")))
        sampled_paths = self._sample_frames(frame_paths)

        frames = []
        for path in sampled_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
            frames.append(img_tensor)

        # Stack separate frame tensors into a single video block
        # Resulting shape: [Frames, Channels, Height, Width]
        video_tensor = torch.stack(frames)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # --- STEP-BY-STEP TESTING BLOCK ---
    print("Initializing DataLoader validation test...")
    
    # Resolve config path cleanly relative to running directory
    config_path = "config.yaml" if os.path.exists("config.yaml") else "../../config.yaml"
    
    try:
        # Initialize training dataset split
        dataset = TennisActionDataset(config_path, split="train")
        print(f"Dataset successfully built! Found {len(dataset)} items across {len(dataset.classes)} action classes.")
        print(f"Class Mapping Index: {dataset.class_to_idx}")

        # Initialize native PyTorch DataLoader
        config = load_config(config_path)
        batch_size = config["training"]["batch_size"]
        num_workers = config["training"]["num_workers"]
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, # Set to 0 for internal testing script safety
            pin_memory=config["training"]["pin_memory"]
        )

        # Draw a single baseline test batch
        videos, labels = next(iter(dataloader))
        
        print("\n=== Verified Pipeline Output Tensors ===")
        print(f"Batch Video Tensor Shape : {videos.shape}") 
        print(f"Batch Labels Tensor Shape: {labels.shape}")
        print(f"Target Labels in Batch   : {labels.tolist()}")
        print("=========================================\n")
        print("Success! Step 2 complete. The tensor pipeline matches target specifications.")

    except Exception as e:
        print(f"\nExecution Error occurred: {e}")
        print("Verify your data/frames/train directory contains non-empty class folders.")