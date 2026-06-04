import os
import yaml
import torch
import torch.nn as nn
from torchvision import models

def load_config(config_path):
    """Loads the global configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_backbone(name="resnet18", pretrained=True):
    """
    Returns a backbone CNN (without final classifier) and its feature dimension.
    """
    name = name.lower()
    
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        feat_dim = 512

    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        backbone.classifier = nn.Identity()
        feat_dim = 1024

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feat_dim = backbone.classifier[1].in_features  # Usually 1280
        backbone.classifier = nn.Identity()

    elif name == "inception_v3":
        # Note: Inception-v3 expects 299x299 inputs. If you switch to this, 
        # ensure frame_size in config.yaml is updated to 299.
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        backbone = models.inception_v3(weights=weights, aux_logits=False)
        backbone.fc = nn.Identity()
        feat_dim = 2048

    else:
        raise ValueError(f"Backbone {name} not supported. Check config.yaml.")

    return backbone, feat_dim

class SpatialExtractor(nn.Module):
    def __init__(self, config_path):
        super(SpatialExtractor, self).__init__()
        self.config = load_config(config_path)
        
        backbone_name = self.config["model"]["backbone"]["name"]
        pretrained = self.config["model"]["backbone"]["pretrained"]
        
        # Initialize the backbone using the factory function
        self.backbone, self.feature_dim = get_backbone(backbone_name, pretrained)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Video sequence of shape [Batch, Frames, Channels, Height, Width]
        Returns:
            torch.Tensor: Feature embeddings of shape [Batch, Frames, Feature_Dim]
        """
        B, F, C, H, W = x.shape
        
        # 2D CNNs cannot process 5D tensors. We fuse Batch and Frames together.
        # Shape becomes: [Batch * Frames, Channels, Height, Width]
        x = x.view(B * F, C, H, W)
        
        # Extract features for every frame independently
        features = self.backbone(x)
        
        # Un-fuse the dimensions to separate Batch and Frames again
        # Shape becomes: [Batch, Frames, Feature_Dim]
        features = features.view(B, F, self.feature_dim)
        
        return features

if __name__ == "__main__":
    # --- STEP-BY-STEP TESTING BLOCK ---
    print("Initializing Spatial Extractor validation test...")
    
    # Resolve config path cleanly
    config_path = "config.yaml" if os.path.exists("config.yaml") else "../../config.yaml"
    
    try:
        # Load the model wrapper
        model = SpatialExtractor(config_path)
        print(f"Model loaded with backbone: {model.config['model']['backbone']['name']}")
        print(f"Output Feature Dimension per frame: {model.feature_dim}")
        
        # Create a dummy tensor that mimics the output of our DataLoader
        # [Batch=2, Frames=64, Channels=3, Height=224, Width=224]
        # Using a small batch of 2 so CPU testing doesn't freeze
        dummy_video_batch = torch.randn(2, 64, 3, 224, 224)
        
        print(f"\nFeeding dummy tensor of shape {dummy_video_batch.shape} into the extractor...")
        
        model.eval()
        with torch.no_grad():
            output_features = model(dummy_video_batch)
            
        print("\n=== Verified Pipeline Output Tensors ===")
        print(f"Output Features Shape: {output_features.shape}")
        print("=========================================\n")
        print("Success! Step 3 complete. The CNN correctly processed the video sequence.")
        
    except Exception as e:
        print(f"\nExecution Error occurred: {e}")