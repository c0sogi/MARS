import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Layer.
    Applies Max Pooling and Min Pooling in parallel and concatenates the results.
    This preserves both the peaks (strong signals) and shadows (weak signals/voids).
    """

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Max Pooling
        out_max = self.max_pool(x)

        # Min Pooling: Implemented as -Max(-x)
        # This effectively finds the minimum values in the window.
        out_min = -F.max_pool2d(-x, self.kernel_size, self.stride, self.padding)

        # Concatenate along channel dimension
        return torch.cat([out_max, out_min], dim=1)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) and a shared MLP.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Ensure hidden channels is at least 1
        hidden_channels = max(in_channels // reduction_ratio, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply MLP to both pooling outputs
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        # Sum and activate
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) across channels and a Convolution layer.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Padding to maintain spatial dimensions
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise Avg Pool
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise Max Pool
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Convolve and activate
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Refine features with Channel Attention
        out = x * self.channel_att(x)
        # Refine features with Spatial Attention
        out = out * self.spatial_att(out)
        return out


class DualPathReadout(nn.Module):
    """
    Robust Dual-Path Readout Interface.
    Processes the feature map through two parallel paths to capture diverse signal properties:
    1. Path A: Spatial Structure (Conv -> Flatten)
    2. Path B: Robust Intensity (Global Avg Pool)

    Cite solution_lesson_node_00127: Avoid computing higher-order global statistics (like Std Dev)
    on deep feature maps with low spatial resolution.
    Cite solution_lesson_node_00123: Decouple spatial structure from global statistics.
    """

    def __init__(self, in_channels, path_a_out_channels=48):
        super(DualPathReadout, self).__init__()

        # Path A: Spatial Structure
        # 3x3 Conv, Padding=1, Stride=1 to reduce channels while keeping spatial dims
        self.path_a_conv = nn.Conv2d(
            in_channels, path_a_out_channels, kernel_size=3, stride=1, padding=1
        )

        # Path B is parameter-free

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]

        # --- Path A: Spatial Structure ---
        # Captures coarse geometry and shape
        feat_a = self.path_a_conv(x)
        feat_a = feat_a.view(feat_a.size(0), -1)  # Flatten

        # --- Path B: Robust Intensity ---
        # Captures global signal strength, robust to noise
        # Cite solution_lesson_node_00115: Use Global Avg Pooling instead of Max Pooling for robustness
        feat_b = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # --- Fusion ---
        # Concatenate features
        out = torch.cat([feat_a, feat_b], dim=1)
        return out


class TriPathReadout(nn.Module):
    """
    Tri-Path Readout Interface.
    Processes the feature map through three parallel paths:
    1. Path A: Spatial Structure (Conv -> Flatten)
    2. Path B: Robust Intensity (Global Avg Pool)
    3. Path C: Texture Variance (Global Std)
    """

    def __init__(self, in_channels, path_a_out_channels=48):
        super(TriPathReadout, self).__init__()
        self.path_a_conv = nn.Conv2d(
            in_channels, path_a_out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # Path A: Spatial Structure
        feat_a = self.path_a_conv(x)
        feat_a = feat_a.view(feat_a.size(0), -1)

        # Path B: Robust Intensity (Global Avg Pool)
        feat_b = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Path C: Texture Variance (Global Std)
        # Calculate standard deviation across spatial dimensions (H, W)
        feat_c = torch.std(x, dim=(2, 3), keepdim=False)

        # Fusion
        out = torch.cat([feat_a, feat_b, feat_c], dim=1)
        return out
