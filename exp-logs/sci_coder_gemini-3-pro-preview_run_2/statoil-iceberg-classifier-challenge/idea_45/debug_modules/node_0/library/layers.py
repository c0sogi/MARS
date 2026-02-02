import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for CBAM.
    Utilizes Mixed Pooling (Average + Max) to refine channel features.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # Ensure hidden planes is at least a reasonable size (e.g., 4) to prevent information bottleneck
        hidden_planes = max(in_planes // ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP implemented as 1x1 Convolutions for efficiency
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply shared MLP to both pooling outputs
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Sum and activate
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for CBAM.
    Utilizes Mixed Pooling (Average + Max) across the channel dimension.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Input channels = 2 (1 for AvgPool, 1 for MaxPool)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average Pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max Pooling along the channel axis
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)

        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.
    Applies Max Pooling (to capture signal peaks) and Min Pooling (to capture shadows)
    in parallel, then concatenates the results along the channel dimension.
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Max Pooling (Peaks)
        max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        # Min Pooling (Shadows)
        # Implemented as -Max(-x)
        min_pool = -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)

        # Concatenate: (B, C, H, W) -> (B, 2C, H', W')
        return torch.cat([max_pool, min_pool], dim=1)


class ContextGating(nn.Module):
    """
    Context-Gating Mechanism.
    Uses a Global Context vector to dynamically modulate Spatial Features.

    Formula: f' = f * alpha + f
    where alpha = Sigmoid(MLP(g))
    """

    def __init__(self, context_dim, feature_dim, hidden_dim=None):
        super(ContextGating, self).__init__()

        # Define hidden dimension for the lightweight MLP
        if hidden_dim is None:
            hidden_dim = max(context_dim // 2, 16)

        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, spatial_features, global_context):
        """
        Args:
            spatial_features (torch.Tensor): Flattened spatial features (Batch, FeatureDim).
            global_context (torch.Tensor): Global context vector (Batch, ContextDim).

        Returns:
            torch.Tensor: Gated spatial features (Batch, FeatureDim).
        """
        # Generate gate vector alpha
        alpha = self.mlp(global_context)

        # Apply Residual Gating
        # f' = f * alpha + f
        out = spatial_features * alpha + spatial_features

        return out
