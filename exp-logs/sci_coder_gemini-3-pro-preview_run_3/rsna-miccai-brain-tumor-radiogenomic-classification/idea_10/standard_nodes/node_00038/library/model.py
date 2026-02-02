import torch
import torch.nn as nn
from torchvision import models


class MGMTNet(nn.Module):
    """
    EfficientNet-B0 adapted for 2.5D Volumetric Input (64 channels).
    Directly modifies the first layer to accept stacked slices.
    Cite solution_lesson_node_00030: Native Input Adaptation.
    Cite solution_lesson_node_00034: Stability of <=64 channel stacking.
    """

    def __init__(self):
        super().__init__()

        # Initialize with ImageNet weights
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 1. Adapt First Layer
        # Input: 16 slices * 4 modalities = 64 channels
        in_channels = 64
        original_conv = self.backbone.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Weight Initialization: Recycle RGB weights by averaging and replicating
        # This preserves the magnitude of activations expected by the pretrained network.
        with torch.no_grad():
            # original_conv.weight shape: (Out, 3, K, K)
            # Average across RGB channels -> (Out, 1, K, K)
            avg_weight = torch.mean(original_conv.weight, dim=1, keepdim=True)
            # Replicate to new channels -> (Out, 64, K, K)
            new_conv.weight[:] = avg_weight.repeat(1, in_channels, 1, 1)

        self.backbone.features[0][0] = new_conv

        # 2. Adapt Classifier Head
        # Replace default classifier with a single Linear layer
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, 256, 256)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
