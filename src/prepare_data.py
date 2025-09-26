import os
import cv2
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "VIDEO_RGB"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FRAME_SIZE = (224, 224)   
FPS = 5                   
SPLIT_RATIOS = (0.7, 0.15, 0.15)  

def extract_frames(video_path, output_dir, fps=FPS, frame_size=FRAME_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps)) if video_fps > 0 else 1

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

def prepare_dataset():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_DIR}")

    classes = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for cls in classes:
        video_files = list((RAW_DATA_DIR / cls).glob("*.avi"))
        train_files, temp_files = train_test_split(video_files, train_size=SPLIT_RATIOS[0], random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), random_state=42)

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split, files in splits.items():
            for video in files:
                video_id = video.stem
                output_dir = PROCESSED_DIR / split / cls / video_id
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_count = extract_frames(video, output_dir)
                print(f"[{split}] {cls}/{video_id}: extracted {frame_count} frames")

if __name__ == "__main__":
    prepare_dataset()
