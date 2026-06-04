"""
Evaluation metrics for temporal video action segmentation.
Includes frame-wise accuracy, F1-score, precision, recall, and IoU.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torch

def compute_accuracy(predictions, targets):
    return np.mean(predictions == targets)

def compute_f1_score(predictions, targets, average='weighted'):
    return f1_score(targets, predictions, average=average, zero_division=0)

def compute_precision_recall(predictions, targets, average='weighted'):
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    return precision, recall

def compute_confusion_matrix(predictions, targets):
    return confusion_matrix(targets, predictions)

def compute_iou(predictions, targets, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = predictions == cls
        target_mask = targets == cls
        
        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)
        
        iou = intersection / union if union > 0 else 0.0
        iou_per_class.append(iou)
    
    return np.array(iou_per_class), np.mean(iou_per_class)

class MetricsTracker:
    """Track and aggregate frame-wise metrics across batches."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_losses = []
    
    def update(self, predictions, targets, loss=None):
        # Flatten [Batch, Frames] into a 1D sequence to evaluate every single frame
        preds = predictions.view(-1).cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions.flatten()
        targs = targets.view(-1).cpu().numpy() if isinstance(targets, torch.Tensor) else targets.flatten()
        
        self.all_preds.extend(preds)
        self.all_targets.extend(targs)
        
        if loss is not None:
            self.all_losses.append(loss)
    
    def get_metrics(self):
        if len(self.all_preds) == 0:
            return {}
        
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        accuracy = compute_accuracy(preds, targets)
        f1 = compute_f1_score(preds, targets, average='macro') # Macro is better for imbalanced actions
        precision, recall = compute_precision_recall(preds, targets, average='macro')
        avg_loss = np.mean(self.all_losses) if self.all_losses else 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'avg_loss': avg_loss
        }