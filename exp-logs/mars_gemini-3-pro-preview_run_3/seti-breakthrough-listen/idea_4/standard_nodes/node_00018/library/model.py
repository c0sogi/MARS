import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class TechnoEfficientNet(nn.Module):
    """
    EfficientNet-B0 modified to accept 6-channel input (stacked cadence).
    Cite solution_lesson_node_00011: Pretrained deep backbones are superior to custom shallow architectures.
    """

    def __init__(self, pretrained=True):
        super(TechnoEfficientNet, self).__init__()

        # Load backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Modify first convolution layer to accept 6 channels
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 6
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights for the new channel
        # We copy the weights from the first 3 channels to the next 3 channels
        # This gives a good initialization assuming On and Off target stats are somewhat similar visually
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight

        self.backbone.features[0][0] = new_conv

        # Modify classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 6, H, W).
        """
        return self.backbone(x)
