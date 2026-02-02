import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.

    This block replaces standard Squeeze-and-Excitation (SE) blocks. Instead of global average pooling,
    it pools features along the horizontal and vertical directions separately. These direction-aware
    features are encoded via a shared convolution and then used to generate attention maps that
    recalibrate the input feature map, preserving spatial texture information critical for
    identifying cactus spines.
    """

    def __init__(self, inp, reduction=32):
        """
        Args:
            inp (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate channel dimension.
        """
        super(CoordinateAttention, self).__init__()

        # Adaptive Average Pooling for H and W directions
        # pool_h: pools width to 1, keeping height dynamic -> (N, C, H, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # pool_w: pools height to 1, keeping width dynamic -> (N, C, 1, W)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Calculate intermediate channels (mip), ensuring a minimum size (e.g., 8)
        mip = max(8, inp // reduction)

        # Shared 1x1 Convolution for encoding spatial information
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # 1x1 Convolutions for generating attention weights for each direction
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the Coordinate Attention block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W) recalibrated by attention.
        """
        identity = x
        n, c, h, w = x.size()

        # 1. Coordinate Information Embedding
        # Pool along width -> (N, C, H, 1)
        x_h = self.pool_h(x)
        # Pool along height -> (N, C, 1, W) -> Permute to (N, C, W, 1) for concatenation
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 2. Coordinate Attention Generation
        # Concatenate along the spatial dimension (H + W) -> (N, C, H+W, 1)
        y = torch.cat([x_h, x_w], dim=2)

        # Encode via shared convolution, BN, and activation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into height and width features
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Permute width features back to (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention maps via 1x1 convolution and Sigmoid
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 3. Re-weighting
        # Broadcast multiplication: (N, C, H, W) * (N, C, H, 1) * (N, C, 1, W)
        out = identity * a_h * a_w
        return out
