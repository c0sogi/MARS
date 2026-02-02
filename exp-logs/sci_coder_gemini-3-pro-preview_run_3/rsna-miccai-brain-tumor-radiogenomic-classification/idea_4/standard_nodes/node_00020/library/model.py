import torch
import torch.nn as nn
from torchvision import models


class EfficientNet25D(nn.Module):
    """
    Early Fusion 2.5D Network for MGMT Promoter Methylation Prediction.

    Cite solution_lesson_node_00018: Replaced Siamese Late Fusion with Early Fusion.
    This model processes all MRI modalities and slices as a single stacked input tensor.
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        in_channels=64,
        num_classes=1,
    ):
        """
        Args:
            backbone_name (str): Name of the backbone architecture. Defaults to 'efficientnet_b0'.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels. Defaults to 64 (4 mods * 16 slices).
            num_classes (int): Number of output classes. Defaults to 1.
        """
        super(EfficientNet25D, self).__init__()

        # 1. Load Backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify First Convolutional Layer
        # Replace 3-channel input with in_channels (64)
        original_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        self.backbone.features[0][0] = new_conv

        # 3. Modify Classifier Head
        # EfficientNet-B0 classifier is a Sequential[Dropout, Linear]
        # We replace the final Linear layer
        original_classifier = self.backbone.classifier

        # The last layer is at index 1
        in_features = original_classifier[1].in_features

        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): (B, 64, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        return self.backbone(x)
