import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, densenet121, inception_v3
from torchvision.models import ResNet18_Weights, DenseNet121_Weights, Inception_V3_Weights

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels=(256, 256), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            padding = (kernel_size - 1) * dilation
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.net(x)
        out = out.transpose(1, 2)
        return out

def get_backbone(name="resnet18", pretrained=True):
    """
    Returns a backbone CNN (without final classifier) and the feature dimension
    Compatible with resnet18, densenet121, inception_v3.
    """
    name = name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        feat_dim = 512

    elif name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)
        backbone.classifier = nn.Identity()
        feat_dim = 1024

    elif name == "inception_v3":
        if pretrained:
            # Use pretrained weights, but ignore aux_logits conflict
            weights = None  # Cannot use pretrained with aux_logits=False
        else:
            weights = None
        backbone = inception_v3(weights=weights, aux_logits=False)
        backbone.fc = nn.Identity()
        feat_dim = 2048

    else:
        raise ValueError(f"Backbone {name} not supported")

    return backbone, feat_dim


class VideoClassifier(nn.Module):
    def __init__(self,
                 backbone_name: str,
                 num_classes: int,
                 temporal: str = "bilstm",
                 pretrained_backbone: bool = True,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 bidirectional: bool = True):
        super().__init__()
        self.backbone, self.feat_dim = get_backbone(backbone_name, pretrained=pretrained_backbone)
        self.temporal = temporal.lower()

        # Temporal module
        if self.temporal == "bilstm":
            self.temporal_module = nn.LSTM(input_size=self.feat_dim,
                                           hidden_size=lstm_hidden,
                                           num_layers=lstm_layers,
                                           batch_first=True,
                                           bidirectional=bidirectional)
            final_dim = lstm_hidden * (2 if bidirectional else 1)

        elif self.temporal == "tcn":
            self.temporal_module = TemporalConvNet(input_size=self.feat_dim)
            final_dim = self.temporal_module.net[-3].out_channels if hasattr(self.temporal_module.net[-3], 'out_channels') else self.feat_dim
            if isinstance(final_dim, tuple) or final_dim is None:
                final_dim = self.feat_dim

        elif self.temporal == "none":
            self.temporal_module = None
            final_dim = self.feat_dim
        else:
            raise ValueError(f"Unsupported temporal module: {temporal}")

        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns logits [B, num_classes]
        """
        B, T, C, H, W = x.shape
        x_frames = x.view(B * T, C, H, W)

        # For InceptionV3, ensure forward returns tensor (not InceptionOutputs)
        if isinstance(self.backbone, type(inception_v3(aux_logits=False))):
            feats = self.backbone(x_frames)
            if hasattr(feats, "logits"):  # just in case
                feats = feats.logits
        else:
            feats = self.backbone(x_frames)

        feats = feats.view(B, T, -1)

        if self.temporal == "none":
            clip_feat = feats.mean(dim=1)
            return self.classifier(clip_feat)

        elif self.temporal == "bilstm":
            lstm_out, _ = self.temporal_module(feats)
            clip_feat = lstm_out.mean(dim=1)
            return self.classifier(clip_feat)

        elif self.temporal == "tcn":
            tcn_out = self.temporal_module(feats)
            clip_feat = tcn_out.mean(dim=1)
            return self.classifier(clip_feat)
