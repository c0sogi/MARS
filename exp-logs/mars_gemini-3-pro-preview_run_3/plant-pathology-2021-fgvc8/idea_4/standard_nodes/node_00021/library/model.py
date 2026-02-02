import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleConvNeXt(nn.Module):
    """
    AppleConvNeXt model implementation based on ConvNeXt-Small.

    This class wraps a timm-based ConvNeXt backbone and explicitly defines
    the classification head to include Layer Normalization before the final
    fully connected layer, as per the architectural requirements.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super(AppleConvNeXt, self).__init__()

        # Load the backbone from timm
        # We use num_classes=0 and global_pool='' to retrieve the raw feature map
        # (B, C, H, W) so we can manually build the head with the required LayerNorm.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            drop_path_rate=Config.DROP_PATH_RATE,
            num_classes=0,
            global_pool="",
        )

        # Get the number of features output by the backbone (e.g., 768 for ConvNeXt-Small)
        self.num_features = self.backbone.num_features

        # Explicitly define the head structure
        # 1. Global Average Pooling (B, C, H, W) -> (B, C, 1, 1)
        # 2. Flatten (B, C, 1, 1) -> (B, C)
        # 3. Layer Normalization for feature stabilization
        # 4. Linear Projection to class logits
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, Config.NUM_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images (B, 3, H, W).

        Returns:
            torch.Tensor: Class logits (B, NUM_CLASSES).
        """
        # Extract spatial features from the backbone
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.head(features)

        return logits
