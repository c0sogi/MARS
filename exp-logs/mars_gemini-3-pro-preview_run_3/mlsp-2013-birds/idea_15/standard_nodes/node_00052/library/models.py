import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    A unified model class for bird species classification using various backbones.
    Implements Concatenated Pooling (GAP + GMP) to better capture sparse audio events.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONES[0],
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            num_classes (int): Number of output classes (species).
            pretrained (bool): Whether to initialize with ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()
        self.backbone_name = backbone_name

        # Create the backbone using timm
        # global_pool='avg' ensures we get the global average pooled vector (B, C)
        # num_classes=0 removes the default classification head.
        # in_chans=3 ensures compatibility with the pseudo-RGB input.
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                in_chans=Config.IN_CHANNELS,
            )
        except Exception as e:
            raise ValueError(f"Failed to create backbone '{backbone_name}': {e}")

        # Determine the number of input features from the backbone
        self.in_features = self.backbone.num_features

        # Standard Global Average Pooling (handled by timm)
        self.pooling_dim = self.in_features

        # Linear Classification Head
        self.fc = nn.Linear(self.pooling_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # 1. Extract features from the backbone
        # Shape: (Batch, Channels) because global_pool="avg"
        features = self.backbone(x)

        # 2. Classification Head
        # Shape: (Batch, Num_Classes)
        logits = self.fc(features)

        return logits
