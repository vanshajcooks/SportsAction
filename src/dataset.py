# src/dataset.py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import glob

class SportsDataset(Dataset):
    def __init__(self, data_root, split="train", num_frames=16, backbone="resnet18"):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.backbone = backbone.lower()

        # Set image size based on backbone
        if self.backbone == "inception_v3":
            self.img_size = 299
        else:
            self.img_size = 224  # ResNet / DenseNet default

        # Define transforms
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Load clip (frame folder) paths and labels
        self.clip_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(os.path.join(self.data_root, split)))
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.data_root, split, cls)
            for clip in os.listdir(cls_dir):
                clip_path = os.path.join(cls_dir, clip)
                if os.path.isdir(clip_path):
                    self.clip_paths.append(clip_path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        label = self.labels[idx]

        try:
            frames = self.load_frames_from_folder(clip_path, self.num_frames)
        except RuntimeError as e:
            print(f"⚠️ Skipping unreadable clip: {clip_path} | Reason: {e}")
            return None

        frames = [self.transform(Image.fromarray(f)) for f in frames]
        clip = np.stack(frames)  # shape: (T, C, H, W)
        return torch.from_numpy(clip).float(), label

    def load_frames_from_folder(self, folder_path, num_frames):
        # Get all image paths in the folder
        img_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if len(img_paths) == 0:
            raise RuntimeError("No frames found in folder")

        # Sample frames evenly
        indices = np.linspace(0, len(img_paths) - 1, num=num_frames, dtype=int)
        frames = [np.array(Image.open(img_paths[i]).convert("RGB")) for i in indices]

        # Pad if less frames than num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

        return frames
