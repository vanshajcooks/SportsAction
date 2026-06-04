# src/utils/visualizer.py
"""
Visualization utilities for video action recognition.
Overlay predictions on video frames and save visualizations.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class VideoVisualizer:
    """Visualize predictions on video frames."""
    
    def __init__(self, class_names: List[str], output_dir: Optional[Path] = None):
        """
        Args:
            class_names: List of action class names
            output_dir: Directory to save visualizations (optional)
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def overlay_prediction(self,
                          frame: np.ndarray,
                          prediction: int,
                          confidence: float,
                          frame_idx: int = 0) -> np.ndarray:
        """
        Overlay prediction on a single frame.
        
        Args:
            frame: (H, W, 3) BGR image
            prediction: Predicted class index
            confidence: Confidence score (0-1)
            frame_idx: Frame number for display
        
        Returns:
            ndarray: Frame with overlay
        """
        frame_copy = frame.copy()
        
        # Get class name and color
        class_name = self.class_names[prediction] if prediction < len(self.class_names) else "Unknown"
        color = self._get_color(prediction)
        
        # Draw background rectangle
        text = f"{class_name} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = 10, 40
        
        cv2.rectangle(frame_copy, (x - 5, y - text_size[1] - 5),
                      (x + text_size[0] + 5, y + 5), color, -1)
        cv2.putText(frame_copy, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # Add frame number
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(frame_copy, frame_text, (10, frame_copy.shape[0] - 10),
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame_copy
    
    def create_video_with_predictions(self,
                                     frames: List[np.ndarray],
                                     predictions: List[int],
                                     confidences: List[float],
                                     output_path: str,
                                     fps: int = 24) -> None:
        """
        Create a video file with predictions overlaid on frames.
        
        Args:
            frames: List of (H, W, 3) BGR frames
            predictions: List of predicted class indices
            confidences: List of confidence scores
            output_path: Path to save video
            fps: Frames per second for output video
        """
        if len(frames) == 0:
            print("No frames to visualize")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(frames):
            pred = predictions[i] if i < len(predictions) else 0
            conf = confidences[i] if i < len(confidences) else 0.0
            
            annotated_frame = self.overlay_prediction(frame, pred, conf, i)
            out.write(annotated_frame)
        
        out.release()
        print(f"Saved visualization to {output_path}")
    
    @staticmethod
    def _get_color(class_idx: int) -> Tuple[int, int, int]:
        """Get a consistent color for each class."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
        ]
        return colors[class_idx % len(colors)]
    
    def save_frame(self, frame: np.ndarray, output_name: str) -> None:
        """
        Save a single annotated frame.
        
        Args:
            frame: (H, W, 3) BGR image
            output_name: Name of output file (e.g., 'frame_001.jpg')
        """
        if self.output_dir is None:
            print("Warning: No output directory specified")
            return
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), frame)
        print(f"Saved frame to {output_path}")
