import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AnisotropicBackbone(nn.Module):
    """
    Anisotropic Backbone based on ResNet-34.

    This backbone wraps a standard ResNet-34 but modifies the downsampling
    strides to be anisotropic. Specifically, it downsamples the height dimension
    normally (factor of 32) but limits the downsampling of the width dimension
    (factor of 2) to preserve horizontal resolution for dense chemical label prediction.

    Input:  (B, 3, 256, W)
    Output: (B, W/2, 512)
    """

    def __init__(self, model_name=Config.ENCODER_NAME, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture (must be 'resnet34').
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(AnisotropicBackbone, self).__init__()

        if model_name != "resnet34":
            raise ValueError(
                f"AnisotropicBackbone currently only supports 'resnet34', got {model_name}"
            )

        # Load the base ResNet34 model
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.net = models.resnet34(weights=weights)

        # ---------------------------------------------------------
        # Stride Modification for Anisotropic Downsampling
        # Target: H downsample 32x, W downsample 2x
        # ---------------------------------------------------------

        # 1. Modify MaxPool stride
        # Original: kernel_size=3, stride=2, padding=1
        # New: stride=(2, 1) -> Downsamples H by 2, W by 1
        self.net.maxpool.stride = (2, 1)

        # 2. Modify Layer Strides
        # In ResNet, downsampling happens in the first block of layers 2, 3, and 4.
        # We modify the stride of the 3x3 conv and the 1x1 downsample projection.
        layers_to_modify = ["layer2", "layer3", "layer4"]

        for layer_name in layers_to_modify:
            layer = getattr(self.net, layer_name)
            # The downsampling block is always the first block (index 0) in the layer
            block = layer[0]

            # Modify the main 3x3 convolution stride
            # Original stride is usually (2, 2)
            if hasattr(block, "conv1"):
                block.conv1.stride = (2, 1)

            # Modify the residual connection 1x1 projection stride
            if block.downsample is not None:
                # block.downsample is a Sequential(Conv2d, BN)
                # We access the Conv2d at index 0
                block.downsample[0].stride = (2, 1)

        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------
        # Remove the classification head components as they are not used
        del self.net.fc
        del self.net.avgpool

    def forward(self, x):
        """
        Forward pass of the backbone.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Feature sequence of shape (Batch, Seq_Len, Channels).
                          Seq_Len corresponds to the width dimension.
        """
        # Stem
        x = self.net.conv1(x)  # (B, 64, H/2, W/2)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)  # (B, 64, H/4, W/2) -> Modified stride (2,1)

        # Layers
        x = self.net.layer1(x)  # (B, 64, H/4, W/2)  -> Stride 1
        x = self.net.layer2(x)  # (B, 128, H/8, W/2) -> Modified stride (2,1)
        x = self.net.layer3(x)  # (B, 256, H/16, W/2)-> Modified stride (2,1)
        x = self.net.layer4(x)  # (B, 512, H/32, W/2)-> Modified stride (2,1)

        # Feature Aggregation
        # Collapse the height dimension by averaging
        # Input: (B, 512, H', W') -> Output: (B, 512, W')
        x = x.mean(dim=2)

        # Permute to (Batch, Sequence, Channels) for Transformer/RNN compatibility
        # Output: (B, W', 512)
        x = x.permute(0, 2, 1)

        return x
