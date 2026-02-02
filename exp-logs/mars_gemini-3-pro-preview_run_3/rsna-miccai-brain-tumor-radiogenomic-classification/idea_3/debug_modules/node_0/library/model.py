import torch
import torch.nn as nn
import timm


class BraTS25DEfficientNet(nn.Module):
    """
    2.5D Convolutional Neural Network based on EfficientNet-B0.

    Architecture:
    - Backbone: EfficientNet-B0 initialized with ImageNet weights.
    - Input: Modified first convolutional layer to accept 64 channels.
      (corresponding to 16 depth slices * 4 MRI modalities).
    - Output: Single logit (before Sigmoid) for binary classification.
    """

    def __init__(self, pretrained=True):
        super(BraTS25DEfficientNet, self).__init__()

        # Load EfficientNet-B0 from timm
        # in_chans=64: Automatically replaces the first conv layer to accept 64 channels.
        #              timm handles weight initialization for the new channels (often by recycling/averaging).
        # num_classes=1: Replaces the classifier head with a Linear layer outputting 1 value.
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, in_chans=64, num_classes=1
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 64, Height, Width).
                              64 channels = 16 slices x 4 modalities.

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1).
        """
        # Pass through the backbone
        # timm efficientnet implementations include the Global Average Pooling
        # and the Flatten operation before the classifier.
        logits = self.backbone(x)

        return logits
