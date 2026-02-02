import torch
import torch.nn as nn
from torchvision import models


class Siamese25DNet(nn.Module):
    """
    Siamese Multi-Stream 2.5D Network for MGMT Promoter Methylation Prediction.

    This model processes 4 MRI modalities (FLAIR, T1w, T1wCE, T2w) as independent streams
    using a shared 2D CNN backbone (EfficientNet-B0). The input to the backbone is a
    32-channel tensor representing 32 depth slices of the MRI volume.

    The features from the 4 streams are concatenated (Late Fusion) and passed through
    a final classification head.
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        in_channels=32,
        num_classes=1,
    ):
        """
        Args:
            backbone_name (str): Name of the backbone architecture. Defaults to 'efficientnet_b0'.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels (slices) per modality. Defaults to 32.
            num_classes (int): Number of output classes. Defaults to 1 (binary classification).
        """
        super(Siamese25DNet, self).__init__()

        # 1. Load Backbone
        weights = "DEFAULT" if pretrained else None
        if backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=weights)
            # EfficientNet-B0 output feature dimension is 1280
            self.feature_dim = 1280
        else:
            # Fallback to EfficientNet-B0 if other names provided for this specific implementation
            self.backbone = models.efficientnet_b0(weights=weights)
            self.feature_dim = 1280

        # 2. Modify First Convolutional Layer
        # The original first layer expects 3 channels (RGB). We replace it to accept 'in_channels' (32).
        # Access the first layer: backbone.features[0][0]
        original_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # 3. Define Fusion Head
        # The backbone's original classifier is ignored. We define a new head that takes
        # the concatenated features from all 4 streams.
        # Input dim = feature_dim * 4 (modalities)
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(self.feature_dim * 4, num_classes)
        )

    def forward_one_stream(self, x):
        """
        Passes a single modality tensor through the shared backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 32, Height, Width).

        Returns:
            torch.Tensor: Feature vector of shape (Batch, feature_dim).
        """
        # Pass through feature extractor (Conv layers)
        x = self.backbone.features(x)

        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        x = self.backbone.avgpool(x)

        # Flatten: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)

        return x

    def forward(self, flair, t1w, t1wce, t2w):
        """
        Forward pass for the Siamese Network.

        Args:
            flair (torch.Tensor): (B, 32, H, W)
            t1w   (torch.Tensor): (B, 32, H, W)
            t1wce (torch.Tensor): (B, 32, H, W)
            t2w   (torch.Tensor): (B, 32, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # 1. Siamese Processing (Shared Weights)
        f_flair = self.forward_one_stream(flair)
        f_t1w = self.forward_one_stream(t1w)
        f_t1wce = self.forward_one_stream(t1wce)
        f_t2w = self.forward_one_stream(t2w)

        # 2. Late Fusion
        # Concatenate the feature vectors from all modalities
        combined_features = torch.cat([f_flair, f_t1w, f_t1wce, f_t2w], dim=1)

        # 3. Classification Head
        logits = self.fusion_head(combined_features)

        return logits
