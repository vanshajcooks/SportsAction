# src/models/video_classifier.py
"""
Main video classification model combining spatial and temporal branches.
"""
import torch
import torch.nn as nn
from .spatial_extractor import get_backbone
from .temporal_brain import TemporalModule


class VideoClassifier(nn.Module):
    """
    Video Classification model using a spatial backbone + temporal module.
    
    Architecture:
    - Spatial Branch: ResNet18, DenseNet121, or Inception-v3 for frame-level features
    - Temporal Branch: BiLSTM, LSTM, GRU, or TCN for temporal modeling
    - Classifier: Linear layer for action classification
    """
    
    def __init__(self,
                 backbone_name: str,
                 num_classes: int,
                 temporal: str = "bilstm",   
                 pretrained_backbone: bool = True,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 bidirectional: bool = True):
        """
        Args:
            backbone_name (str): Spatial backbone - 'resnet18', 'densenet121', 'inception_v3'
            num_classes (int): Number of action classes
            temporal (str): Temporal module - 'bilstm', 'lstm', 'gru', 'tcn', 'none'
            pretrained_backbone (bool): Use pretrained ImageNet weights
            lstm_hidden (int): Hidden dimension for RNN modules
            lstm_layers (int): Number of RNN layers
            bidirectional (bool): Use bidirectional RNN (for BiLSTM/LSTM/GRU)
        """
        super().__init__()
        
        # Spatial feature extraction
        self.backbone, self.feat_dim = get_backbone(backbone_name, pretrained=pretrained_backbone)
        
        # Temporal modeling
        temporal_module, output_dim = TemporalModule.create(
            temporal,
            input_size=self.feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional and temporal != "none"
        )
        self.temporal_module = temporal_module
        self.temporal_type = temporal.lower()
        
        # Classification head
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch, time, channels, height, width
        Returns:
            logits: (B, num_classes) - class scores
        """
        B, T, C, H, W = x.shape
        
        # Extract frame-level features using backbone
        x_frames = x.view(B * T, C, H, W)
        feats = self.backbone(x_frames)       # → (B*T, feat_dim)
        feats = feats.view(B, T, -1)          # → (B, T, feat_dim)

        # Apply temporal modeling
        if self.temporal_type in ["bilstm", "lstm", "gru"]:
            out, _ = self.temporal_module(feats)   # → (B, T, hidden_dim)
            clip_feat = out.mean(dim=1)            # Temporal average pooling
            
        elif self.temporal_type == "tcn":
            tcn_out = self.temporal_module(feats)  # → (B, T, output_dim)
            clip_feat = tcn_out.mean(dim=1)        # Temporal average pooling
            
        elif self.temporal_type == "none":
            clip_feat = feats.mean(dim=1)          # Direct spatial averaging
            
        else:
            raise ValueError(f"Unknown temporal type: {self.temporal_type}")

        # Classification
        logits = self.classifier(clip_feat)
        return logits
