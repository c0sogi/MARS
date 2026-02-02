import torch
import torch.nn as nn
import timm
from library.config import Config


class VolumetricEfficientNet(nn.Module):
    """
    Volumetric EfficientNet-B0 with Grouped Convolutional Stem.
    Processes a single 2.5D ROI stack per patient (Cite 00005).

    Architecture:
    1. Input: (Batch, 12, H, W)
    2. Backbone: EfficientNet-B0 with modified stem (Groups=4).
    3. Classifier: Linear layer.
    """

    def __init__(self):
        super(VolumetricEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        # num_classes=0 returns the global pool features (flat vector)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # 2. Modify Stem for 12 Channels + Grouped Conv (Cite 00007)
        # Original stem: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        old_stem = self.backbone.conv_stem

        # Validate assumptions about the backbone
        assert old_stem.in_channels == 3, "Backbone expected to have 3 input channels"

        # Create new stem
        # We use groups=4 to isolate modalities (3 channels per group)
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
        # PyTorch Grouped Conv Weight Shape: (Out, In/Groups, K, K)
        # Old Shape: (32, 3, 3, 3)
        # New Shape: (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        # The shapes match perfectly. We copy the pretrained RGB weights directly.
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight)
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        # Replace the stem in the backbone
        self.backbone.conv_stem = new_stem

        # 4. Classifier
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Extract features using the backbone
        # Output shape: (Batch, Feature_Dim)
        features = self.backbone(x)

        # Classification
        # (Batch, Num_Classes)
        logits = self.classifier(features)

        return logits
