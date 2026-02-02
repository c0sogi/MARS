import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.
    Captures long-range dependencies along spatial directions (H and W)
    by encoding channel-wise information into two 1D feature maps.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the intermediate channel dimension.
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Ensure intermediate dimension is at least 8 to preserve information
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        identity = x
        n, c, h, w = x.size()

        # Pool spatial directions
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Permute to (N, C, W, 1)

        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Shared 1x1 Convolution
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into H and W components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Permute back to (N, C, 1, W)

        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_h * a_w
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Ensure bottleneck dimension is at least 1
        mid_channels = max(1, in_channels // reduction)

        # Using Conv2d 1x1 is equivalent to Linear but avoids reshaping
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Aggregates a sequence of feature vectors into a single vector using
    learnable attention weights.

    Args:
        input_dim (int): Dimension of the input feature vectors.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Features).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Features).
        """
        # Calculate attention scores: (Batch, Time, 1)
        scores = self.attention_weights(x)

        # Normalize scores over the time dimension: (Batch, Time, 1)
        weights = F.softmax(scores, dim=1)

        # Compute weighted sum: (Batch, Features)
        out = torch.sum(x * weights, dim=1)

        return out
