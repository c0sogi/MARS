import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class HRNetSegmenter(nn.Module):
    """
    HRNet-W18 based Semantic Segmenter for Contrail Detection.

    This model utilizes a High-Resolution Network (HRNet) backbone to maintain
    high-resolution representations throughout the forward pass, which is critical
    for detecting thin, linear structures like contrails.

    Architecture:
    - Backbone: HRNet-W18 (pretrained on ImageNet)
    - Head: 1x1 Convolution on the high-resolution output (stride 4)
    - Upsampling: Bilinear upsampling to input resolution
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(HRNetSegmenter, self).__init__()

        # Create HRNet backbone
        # features_only=True returns a list of feature maps from different stages
        # in_chans=3 corresponds to the Ash False-Color Composite
        self.backbone = timm.create_model(
            Config.MODEL_NAME,  # 'hrnet_w18'
            pretrained=pretrained,
            features_only=True,
            in_chans=3,
        )

        # Determine the number of channels in the high-resolution feature map.
        # HRNet outputs a list of features. The first element (index 0) corresponds
        # to the highest resolution stream (stride 4).
        # For hrnet_w18, this is typically 18 channels.
        if hasattr(self.backbone, "feature_info"):
            # feature_info contains metadata about the output channels
            in_channels = self.backbone.feature_info[0]["num_chs"]
        else:
            # Fallback: Perform a dummy forward pass to determine channel count
            dummy_input = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                features = self.backbone(dummy_input)
            in_channels = features[0].shape[1]

        # Lightweight Segmentation Head
        # A simple 1x1 convolution maps the feature dimension to the number of classes (1)
        self.cls_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images, shape (B, 3, H, W).

        Returns:
            torch.Tensor: Raw logits, shape (B, 1, H, W).
        """
        input_size = x.shape[-2:]  # (H, W)

        # Pass through backbone
        # features is a list of tensors: [stride4, stride8, stride16, stride32]
        features = self.backbone(x)

        # Select the high-resolution feature map (stride 4)
        # This stream preserves the spatial precision needed for pixel-level masks
        x_high_res = features[0]

        # Apply segmentation head
        logits = self.cls_head(x_high_res)

        # Upsample to original input resolution
        # HRNet output is 1/4 scale, so we upsample to match input size
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return logits
