import torch
import torch.nn as nn
import timm
from library.config import Config


class ConvNeXtTinyCustom(nn.Module):
    """
    ConvNeXt-Tiny architecture adapted for small resolution histopathology patches.

    This model modifies the standard ConvNeXt stem (4x4 conv, stride 4) to a
    finer-grained 3x3 conv with stride 1. This prevents aggressive downsampling
    at the input stage, preserving spatial information for the 48x48 input crops.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        """
        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights for the backbone.
        """
        super().__init__()

        # Load ConvNeXt-Tiny backbone
        # num_classes=0 ensures we get the pooled feature vector (Identity head)
        # global_pool='avg' is the default for ConvNeXt in timm
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # ---------------------------------------------------------------------
        # Stem Modification
        # ---------------------------------------------------------------------
        # The original stem is: Sequential(Conv2d(3, dims[0], k=4, s=4), LayerNorm(...))
        # We access the first layer (Conv2d) of the stem
        original_conv = self.backbone.stem[0]

        # Create a new convolution layer
        # Kernel: 3x3, Stride: 1, Padding: 1
        # This preserves the input spatial resolution (H, W) -> (H, W)
        new_conv = nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=original_conv.bias is not None,
        )

        # Replace the original stem convolution with the new one
        # The new layer is initialized with random weights by default (PyTorch init)
        # The rest of the backbone retains the pretrained ImageNet weights
        self.backbone.stem[0] = new_conv

        # ---------------------------------------------------------------------
        # Classifier Head
        # ---------------------------------------------------------------------
        # Get the feature dimension of the backbone (768 for convnext_tiny)
        self.num_features = self.backbone.num_features

        # Binary classification head
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Extract features using the modified backbone
        # Shape: (B, num_features)
        features = self.backbone(x)

        # Pass through the classifier
        # Shape: (B, 1)
        logits = self.fc(features)

        return logits
