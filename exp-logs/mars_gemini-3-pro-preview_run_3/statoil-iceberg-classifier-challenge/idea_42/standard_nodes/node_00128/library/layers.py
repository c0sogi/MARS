import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaxBlurPool2d(nn.Module):
    """
    Implements Anti-Aliased Max Pooling.

    Decouples standard MaxPool2d into:
    1. Max Pooling with stride 1 (to preserve high-frequency peaks).
    2. Low-pass filtering (Blur) with stride 2 (for anti-aliased subsampling).

    Reference: "Making Convolutional Networks Shift-Invariant Again", Richard Zhang, 2019.
    """

    def __init__(self, in_channels, kernel_size=3):
        super(MaxBlurPool2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Generate 1D Gaussian kernel
        if kernel_size == 3:
            b = np.array([1.0, 2.0, 1.0])
        elif kernel_size == 5:
            b = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            # Binomial coefficients for general kernel size
            b = np.poly1d([0.5, 0.5]) ** (kernel_size - 1)
            b = b.coeffs

        # Normalize
        b = b / np.sum(b)

        # Create 2D kernel via outer product
        bk = b[:, None] * b[None, :]
        bk = torch.from_numpy(bk.astype(np.float32))

        # Reshape for depthwise convolution: (channels, 1, k, k)
        # groups = in_channels, so weights shape is (in_channels, 1, k, k)
        bk = bk.view(1, 1, kernel_size, kernel_size)
        bk = bk.repeat(in_channels, 1, 1, 1)

        # Register as a buffer so it is part of state_dict but not a parameter
        self.register_buffer("blur_kernel", bk)

        # Calculate padding to maintain spatial consistency
        # We want to emulate the spatial reduction of a stride-2 pooling
        self.pad = (kernel_size - 1) // 2

    def forward(self, x):
        # 1. Max Pooling (Stride 1)
        # Extracts local peaks without downsampling
        x = F.max_pool2d(x, kernel_size=2, stride=1)

        # 2. Blur (Stride 2)
        # Low-pass filter followed by subsampling
        x = F.conv2d(
            x, self.blur_kernel, stride=2, padding=self.pad, groups=self.in_channels
        )

        return x


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module.

    Enhances channel interdependencies.
    - Squeeze: Global Average Pooling.
    - Excitation: Adaptive recalibration via MLP.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        # Ensure hidden dimension is at least 1
        hidden_dim = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        y = self.avg_pool(x).view(b, c)
        # Excitation: (b, c) -> (b, c) -> (b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)
