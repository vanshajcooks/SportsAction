import os
import yaml
import torch
import torch.nn as nn

def load_config(config_path):
    """Loads the global configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

class TemporalConvNet(nn.Module):
    """Multi-scale Temporal Convolution Network (MS-TCN)."""
    
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            
            # Corrected padding formula to guarantee output_length == input_length
            padding = dilation * (kernel_size - 1) // 2 
            
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Channels)
        Returns:
            out: (Batch, Time, Channels)
        """
        x = x.transpose(1, 2)  # Conv1d expects (Batch, Channels, Time)
        out = self.net(x)
        out = out.transpose(1, 2)  # Transpose back to (Batch, Time, Channels)
        return out


class TemporalBrain(nn.Module):
    """Main wrapper that integrates the temporal module with the config and final classifier."""
    
    def __init__(self, config_path, input_dim=1280, num_classes=12):
        super().__init__()
        self.config = load_config(config_path)
        
        temp_cfg = self.config["model"]["temporal"]
        self.temporal_type = temp_cfg["type"].lower()
        hidden_size = temp_cfg["hidden_size"]
        num_layers = temp_cfg["num_layers"]
        bidirectional = temp_cfg.get("bidirectional", False)
        dropout = self.config["model"]["classification"]["dropout"]
        
        # 1. Instantiate the Temporal Module based on config
        if self.temporal_type in ["tcn", "ms-tcn"]:
            num_channels = [hidden_size] * num_layers
            self.temporal = TemporalConvNet(input_dim, num_channels, dropout=dropout)
            out_dim = hidden_size
            
        elif self.temporal_type in ["bilstm", "lstm", "gru"]:
            RNNClass = nn.LSTM if "lstm" in self.temporal_type else nn.GRU
            is_bidir = bidirectional or ("bi" in self.temporal_type)
            
            self.temporal = RNNClass(
                input_size=input_dim, 
                hidden_size=hidden_size,
                num_layers=num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=is_bidir
            )
            out_dim = hidden_size * 2 if is_bidir else hidden_size
            
        elif self.temporal_type == "none":
            self.temporal = nn.Identity()
            out_dim = input_dim
            
        else:
            raise ValueError(f"Unsupported temporal module: {self.temporal_type}")
            
        # 2. Final Classification Layer
        # nn.Linear automatically applies to the last dimension, preserving [Batch, Frames]
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Features of shape [Batch, Frames, Feature_Dim]
        Returns:
            torch.Tensor: Predictions of shape [Batch, Frames, Num_Classes]
        """
        if self.temporal_type in ["tcn", "ms-tcn", "none"]:
            out = self.temporal(x)
        else:
            # RNNs return a tuple: (output, hidden_states)
            out, _ = self.temporal(x)
            
        # Map the temporal features to our specific tennis action classes
        return self.classifier(out)


if __name__ == "__main__":
    # --- STEP-BY-STEP TESTING BLOCK ---
    print("Initializing Temporal Brain validation test...")
    
    config_path = "config.yaml" if os.path.exists("config.yaml") else "../../config.yaml"
    
    try:
        # 1280 corresponds to EfficientNet-B0; 12 corresponds to THETIS classes
        model = TemporalBrain(config_path, input_dim=1280, num_classes=12)
        print(f"Temporal Model loaded: {model.temporal_type.upper()}")
        
        # Shape: [Batch=2, Frames=64, Feature_Dim=1280]
        dummy_features = torch.randn(2, 64, 1280)
        print(f"\nFeeding dummy features of shape {dummy_features.shape}...")
        
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_features)
            
        print("\n=== Verified Pipeline Output Tensors ===")
        print(f"Final Prediction Shape: {predictions.shape}")
        print("=========================================\n")
        print("Success! Step 4 complete. Run this to verify the shape is exactly [2, 64, 12].")
        
    except Exception as e:
        print(f"\nExecution Error occurred: {e}")