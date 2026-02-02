import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config
from library.layers import CoordinateAttention


class TimePreservingResNet18(nn.Module):
    """
    A ResNet-18 backbone modified for:
    1. 1-Channel Audio Input (Spectrogram).
    2. Time-Preserving Strides (Asymmetric) in deeper layers to maintain temporal resolution.
    3. Coordinate Attention blocks after each stage.
    4. Hierarchical Feature Extraction (exposing Layers 2, 3, 4).
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        super(TimePreservingResNet18, self).__init__()

        # Load standard ResNet18
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.resnet = resnet18(weights=weights)

        # 1. Modify Input Layer for 1-channel input
        # Standard ResNet: Conv2d(3, 64, kernel=7, stride=2, padding=3)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        # Initialize the 1-channel weights by averaging the pretrained RGB weights
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight.data = original_conv1.weight.data.mean(
                    dim=1, keepdim=True
                )

        # 2. Modify Strides for Time Preservation
        # Config.RESNET_STRIDES = [(2, 2), (2, 1), (2, 1)] for Layers 2, 3, 4
        # Layer 1 retains default stride (1, 1)

        self._modify_stride(self.resnet.layer2, Config.RESNET_STRIDES[0])
        self._modify_stride(self.resnet.layer3, Config.RESNET_STRIDES[1])
        self._modify_stride(self.resnet.layer4, Config.RESNET_STRIDES[2])

        # 3. Coordinate Attention Modules
        # Placed after each ResNet Layer
        self.ca1 = CoordinateAttention(64)
        self.ca2 = CoordinateAttention(128)
        self.ca3 = CoordinateAttention(256)
        self.ca4 = CoordinateAttention(512)

    def _modify_stride(self, layer, stride):
        """
        Modifies the stride of the first block in a ResNet layer.
        Handles both the main convolution and the downsample path.
        """
        # The first block in the Sequential container handles the stride
        block = layer[0]

        # Modify the main 3x3 convolution stride
        block.conv1.stride = stride

        # Modify the downsample 1x1 convolution stride if it exists
        if block.downsample is not None:
            # downsample is Sequential(Conv2d, BatchNorm)
            block.downsample[0].stride = stride

    def forward(self, x):
        """
        Forward pass returning hierarchical features.

        Args:
            x (torch.Tensor): Input spectrogram (N, 1, F, T)

        Returns:
            list[torch.Tensor]: List of feature maps from [Layer2, Layer3, Layer4]
        """
        # Stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Layer 1 (64 channels)
        x = self.resnet.layer1(x)
        x = self.ca1(x)
        # We don't use Layer 1 for fusion in this configuration, but it's part of the flow

        # Layer 2 (128 channels)
        x = self.resnet.layer2(x)
        x = self.ca2(x)
        feat_l2 = x

        # Layer 3 (256 channels)
        x = self.resnet.layer3(x)
        x = self.ca3(x)
        feat_l3 = x

        # Layer 4 (512 channels)
        x = self.resnet.layer4(x)
        x = self.ca4(x)
        feat_l4 = x

        # Return features for Adaptive Spectral Fusion
        return [feat_l2, feat_l3, feat_l4]
