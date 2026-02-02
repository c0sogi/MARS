import torch
import torch.nn as nn
import timm
from library.config import Config


class MILEfficientNet(nn.Module):
    """
    Volumetric EfficientNet-B0 with Grouped Convolutional Stem.
    Simplified from MIL to Single-Instance (2.5D) based on Lesson 00016.

    Architecture:
    1. Input: (Batch, 12, H, W)
    2. Backbone: EfficientNet-B0 (Modified Stem)
    3. Dropout: Added for regularization (Lesson 00017)
    4. Classifier: Linear layer
    """

    def __init__(self):
        super(MILEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # 2. Modify Stem for 12 Channels + Grouped Conv (Cite 00007)
        old_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
            groups=Config.GROUPS,
        )

        # 3. Initialize Weights (Cite 00006)
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight)
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        self.backbone.conv_stem = new_stem

        # 4. Classifier Head with Dropout (Cite 00017)
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Channels, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Extract features
        features = self.backbone(x)

        # Apply Dropout and Classify
        logits = self.classifier(self.dropout(features))

        return logits
