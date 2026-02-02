import torch
import torch.nn as nn
import timm
from library import config


class SiameseEfficientNet(nn.Module):
    """
    Siamese Hybrid-Sampling 2.5D Network.

    Architecture:
    1. Shared Backbone: EfficientNet-B0
    2. Input Stem: Modified to accept 12 channels (3 slices * 4 modalities)
       using Grouped Convolutions (groups=4) to learn modality-specific features.
    3. Fusion: Concatenates global features from ROI View and Geometric View.
    4. Head: Fully connected layer for binary classification.
    """

    def __init__(self, backbone_name=config.BACKBONE, pretrained=True):
        super(SiameseEfficientNet, self).__init__()

        # 1. Load Shared Backbone
        # Initialize with in_chans=3 to load standard pre-trained RGB weights.
        # num_classes=0 ensures we get the global pooled feature vector.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, in_chans=3
        )

        # 2. Modify Input Stem for 12 Channels & Grouped Convolutions
        # We need to replace the first layer to handle our 12-channel input.
        original_stem = self.backbone.conv_stem

        # Capture original configuration
        out_channels = original_stem.out_channels
        kernel_size = original_stem.kernel_size
        stride = original_stem.stride
        padding = original_stem.padding
        bias = original_stem.bias is not None

        # Create new stem
        # in_channels = 12 (config.IN_CHANNELS)
        # groups = 4 (One group per modality: FLAIR, T1w, T1wCE, T2w)
        # Weight shape for groups=4: (out_channels, in_channels/groups, k, k) -> (32, 3, 3, 3)
        # This shape is identical to the original RGB weights (32, 3, 3, 3).
        self.new_stem = nn.Conv2d(
            in_channels=config.IN_CHANNELS,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=4,
            bias=bias,
        )

        # Initialize weights
        # We copy the pre-trained RGB weights directly.
        # This effectively assigns the pre-trained filters to the different modality groups.
        with torch.no_grad():
            self.new_stem.weight.copy_(original_stem.weight)
            if bias:
                self.new_stem.bias.copy_(original_stem.bias)

        # Replace the stem in the backbone
        self.backbone.conv_stem = self.new_stem

        # 3. Define Fusion Head
        # The backbone outputs a feature vector (e.g., 1280 dim for B0).
        # We concatenate two such vectors (ROI + Geometric).
        self.num_features = self.backbone.num_features

        self.fusion_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.num_features * 2, config.NUM_CLASSES)
        )

    def forward_one(self, x):
        """
        Passes a single view through the shared backbone.
        """
        return self.backbone(x)

    def forward(self, roi_x, geo_x):
        """
        Siamese Forward Pass.

        Args:
            roi_x (torch.Tensor): Content-Adaptive (ROI) View [B, 12, H, W]
            geo_x (torch.Tensor): Geometric View [B, 12, H, W]

        Returns:
            logits (torch.Tensor): Raw output scores [B, 1]
        """
        # 1. Extract features using shared backbone
        feat_roi = self.forward_one(roi_x)  # Shape: [B, num_features]
        feat_geo = self.forward_one(geo_x)  # Shape: [B, num_features]

        # 2. Fuse features
        combined = torch.cat(
            [feat_roi, feat_geo], dim=1
        )  # Shape: [B, num_features * 2]

        # 3. Predict
        logits = self.fusion_head(combined)

        return logits
