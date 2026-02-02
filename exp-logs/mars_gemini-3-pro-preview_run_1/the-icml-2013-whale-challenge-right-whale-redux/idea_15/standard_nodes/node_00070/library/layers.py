import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block.

    Captures long-range dependencies along one spatial direction while preserving
    precise positional information along the other. It decomposes the input feature
    map into two 1D feature encoding processes.

    Reference: "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, in_channels, reduction=32):
        """
        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate channel dimension.
        """
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Attention-weighted tensor of shape (B, C, H, W).
        """
        identity = x
        n, c, h, w = x.size()

        # Feature encoding along two directions
        x_h = self.pool_h(x)  # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1)

        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)

        # Shared transformation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into two tensors
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W)

        # Generate attention maps
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        # Apply attention
        out = identity * a_h * a_w
        return out


class SpecFPN(nn.Module):
    """
    Spectrogram Feature Pyramid Network (SpecFPN).

    Fuses features from ResNet layers 2, 3, and 4 via a top-down pathway.
    Crucially, it restores frequency resolution by upsampling only the frequency
    dimension, respecting the asymmetric strides of the backbone.
    """

    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list[int]): List of input channel counts for [Layer2, Layer3, Layer4].
                                          Example for ResNet18: [128, 256, 512].
            out_channels (int): Number of output channels for the fused features.
        """
        super(SpecFPN, self).__init__()

        # Lateral connections (1x1 convs) to project inputs to common channel dim
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list]
        )

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): List of feature maps [c2, c3, c4] from the backbone.
                                   c2 corresponds to Layer 2, c3 to Layer 3, c4 to Layer 4.

        Returns:
            Tensor: The fused feature map P2, which has the high semantic value of deep layers
                    and the high frequency resolution of early layers.
        """
        if len(inputs) != 3:
            raise ValueError(f"SpecFPN expects 3 inputs, got {len(inputs)}")

        # Unpack inputs (c2 is shallowest/highest res, c4 is deepest/lowest res)
        c2, c3, c4 = inputs

        # 1. Project all features to common channel dimension
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3)
        p2 = self.lateral_convs[0](c2)

        # 2. Top-down fusion with Asymmetric Upsampling

        # Fuse P4 into P3
        # L4 has stride (2,1) relative to L3 (Frequency halved, Time preserved).
        # We upsample Frequency by 2x.
        p4_up = F.interpolate(p4, scale_factor=(2, 1), mode="nearest")

        # Handle potential shape mismatch due to padding/odd dimensions
        if p4_up.shape[2:] != p3.shape[2:]:
            p4_up = F.interpolate(p4, size=p3.shape[2:], mode="nearest")

        p3 = p3 + p4_up

        # Fuse P3 into P2
        # L3 has stride (2,1) relative to L2.
        # We upsample Frequency by 2x.
        p3_up = F.interpolate(p3, scale_factor=(2, 1), mode="nearest")

        if p3_up.shape[2:] != p2.shape[2:]:
            p3_up = F.interpolate(p3, size=p2.shape[2:], mode="nearest")

        p2 = p2 + p3_up

        return p2


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.

    Aggregates a sequence of features into a single vector using a learned
    attention mechanism. This allows the model to focus on the most relevant
    time steps (e.g., the whale call) and ignore noise.
    """

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): Dimensionality of the input features.
        """
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input sequence of shape (Batch, Time, Features).

        Returns:
            Tensor: Aggregated embedding of shape (Batch, Features).
        """
        # Calculate attention scores
        # x: (B, T, C) -> scores: (B, T, 1)
        scores = self.attention(x)

        # Normalize scores to weights
        weights = F.softmax(scores, dim=1)

        # Weighted sum over time
        # (B, T, C) * (B, T, 1) -> (B, T, C) -> sum(dim=1) -> (B, C)
        out = torch.sum(x * weights, dim=1)

        return out
