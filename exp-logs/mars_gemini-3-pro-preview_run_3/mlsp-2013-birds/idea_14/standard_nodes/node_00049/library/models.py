import torch
import torch.nn as nn
import timm
from library.config import Config


class ConcatPoolingHead(nn.Module):
    """
    Concatenates Global Average Pooling and Global Max Pooling.
    GAP captures background/context.
    GMP captures peak activation (strongest bird call signal).
    """

    def __init__(self):
        super(ConcatPoolingHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)

        # Average pooling -> (Batch, Channels, 1, 1) -> (Batch, Channels)
        avg_feat = self.avg_pool(x).flatten(1)

        # Max pooling -> (Batch, Channels, 1, 1) -> (Batch, Channels)
        max_feat = self.max_pool(x).flatten(1)

        # Concatenate: (Batch, 2 * Channels)
        return torch.cat([avg_feat, max_feat], dim=1)


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a specified backbone and ConcatPooling.
    """

    def __init__(self, backbone_name, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm backbone (e.g., 'resnet18').
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(BirdClassifier, self).__init__()

        # Create the backbone
        # num_classes=0 and global_pool='' ensures we get spatial feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the number of output channels from the backbone
        if hasattr(self.backbone, "num_features"):
            self.n_features = self.backbone.num_features
        else:
            # Fallback: run a dummy forward pass to infer shape
            with torch.no_grad():
                dummy_input = torch.zeros(1, Config.IN_CHANNELS, *Config.IMG_SIZE)
                features = self.backbone(dummy_input)
                self.n_features = features.shape[1]

        # Pooling Head
        self.pooling = ConcatPoolingHead()

        # Classifier Head
        # Input dimension is doubled because of ConcatPooling (GAP + GMP)
        self.fc = nn.Linear(self.n_features * 2, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
        Returns:
            torch.Tensor: Logits (B, NumClasses)
        """
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Pool features: (B, 2*C)
        pooled_features = self.pooling(features)

        # Classification logits: (B, NumClasses)
        logits = self.fc(pooled_features)

        return logits
