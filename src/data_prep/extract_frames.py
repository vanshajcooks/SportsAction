import os
import cv2
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_config(config_path):
    """Loads the global configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def extract_frames(video_path, output_dir, target_fps, frame_size):
    """Extracts and resizes frames from a video at a specific FPS."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback if cv2 fails to read FPS metadata
    if video_fps <= 0 or video_fps is None:
        video_fps = 30.0 
        
    frame_interval = max(1, int(round(video_fps / target_fps)))

    frame_count, saved_count = 0, 0
    success, frame = cap.read()
    
    while success:
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, frame_size)
            frame_file = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            saved_count += 1
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    return saved_count

def prepare_dataset(config_path):
    """Reads config, splits dataset, and extracts frames into train/val/test folders."""
    config = load_config(config_path)
    
    # Extract settings from config
    raw_dir = Path(config["data"]["raw_dir"])
    frames_dir = Path(config["data"]["frames_dir"])
    target_fps = config["data"]["fps"]
    frame_size_val = config["data"]["frame_size"]
    frame_size = (frame_size_val, frame_size_val)
    split_ratios = config["data"]["split_ratios"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_dir}")

    # Find action classes (assuming each subfolder in raw_dir is a class)
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    print(f"Target FPS: {target_fps} | Frame Size: {frame_size}")

    for cls in classes:
        print(f"\nProcessing class: {cls}...")
        video_files = list((raw_dir / cls).glob("*.avi"))
        
        if not video_files:
            print(f"No .avi videos found in {cls}. Skipping.")
            continue

        # Train/Val/Test Split calculation
        train_files, temp_files = train_test_split(
            video_files, train_size=split_ratios[0], random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), 
            random_state=42
        )

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        # Extract frames for each split
        for split, files in splits.items():
            for video in tqdm(files, desc=f"[{split}] {cls}"):
                video_id = video.stem
                output_dir = frames_dir / split / cls / video_id
                output_dir.mkdir(parents=True, exist_ok=True)
                
                extract_frames(video, output_dir, target_fps, frame_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from THETIS dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to the global config.yaml file"
    )
    args = parser.parse_args()

    # Ensure config path is resolved correctly whether run from root or src/data_prep/
    config_file = Path(args.config)
    if not config_file.exists():
        # Try looking up two directories if run directly from inside src/data_prep/
        config_file = Path("../../config.yaml")
        if not config_file.exists():
            print("Error: config.yaml not found. Please run from the project root.")
            exit(1)

    prepare_dataset(config_file)