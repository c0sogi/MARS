import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CovariancePooling(nn.Module):
    """
    Global Covariance Pooling with Matrix Power Normalization (Newton-Schulz).
    Extracts second-order statistics (texture) from feature maps.
    """

    def __init__(self, num_features, num_iterations=3):
        super(CovariancePooling, self).__init__()
        self.num_features = num_features
        self.num_iterations = num_iterations

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        N = H * W

        # Reshape to (B, C, N)
        x = x.view(B, C, N)

        # Center the features (subtract mean)
        mean = x.mean(dim=2, keepdim=True)
        x = x - mean

        # Compute Covariance Matrix: (B, C, C)
        # Formula: 1/(N-1) * X * X^T
        # Add epsilon for numerical stability
        cov = torch.bmm(x, x.transpose(1, 2)) / (N - 1 + 1e-5)

        # Matrix Power Normalization via Newton-Schulz Iteration
        # Approximates A^(1/2)

        # 1. Trace Normalization to ensure convergence
        trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(2)
        cov_norm = cov / (trace + 1e-5)

        # 2. Iteration
        # Initialization: Y_0 = A, Z_0 = I
        Y = cov_norm
        Z = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)

        for _ in range(self.num_iterations):
            # T = 3I - Z_k * Y_k
            T = 3 * torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(
                0
            ) - torch.bmm(Z, Y)
            # Y_{k+1} = 0.5 * Y_k * T
            Y = 0.5 * torch.bmm(Y, T)
            # Z_{k+1} = 0.5 * T * Z_k
            Z = 0.5 * torch.bmm(T, Z)

        # Extract Upper Triangular part to remove redundancy
        # The diagonal is included.
        triu_indices = torch.triu_indices(C, C, device=x.device)
        # Advanced indexing returns flattened vector of selected elements
        out = Y[:, triu_indices[0], triu_indices[1]]

        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    """

    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """
    Standard Residual Block with optional SE module.
    """

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=Config.SE_REDUCTION)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class NarrowSEResNet(nn.Module):
    """
    Custom Narrow SE-ResNet with Multi-Scale Global Covariance Pooling.
    """

    def __init__(self):
        super(NarrowSEResNet, self).__init__()

        channels = Config.BLOCK_CHANNELS  # Expected: [16, 32, 64]

        # Initial Convolution: 3 -> 16, 32x32
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 16 channels, 32x32 (No downsampling)
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1)

        # Stage 2: 32 channels, 16x16 (Downsampling)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2)

        # Stage 3: 64 channels, 8x8 (Downsampling)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2)

        # Pooling Layers (Covariance Pooling)
        self.pool2 = CovariancePooling(channels[1])
        self.pool3 = CovariancePooling(channels[2])

        # Calculate output dimension for the linear layer
        # Size of upper triangular part: C * (C + 1) / 2
        dim2 = (channels[1] * (channels[1] + 1)) // 2
        dim3 = (channels[2] * (channels[2] + 1)) // 2
        total_dim = dim2 + dim3

        self.fc = nn.Linear(total_dim, 1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride, blocks=2):
        layers = []
        layers.append(
            ResBlock(in_channels, out_channels, stride=stride, use_se=Config.USE_SE)
        )
        for _ in range(1, blocks):
            layers.append(
                ResBlock(out_channels, out_channels, stride=1, use_se=Config.USE_SE)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)  # Output: (B, 16, 32, 32)

        # Stage 2
        f2 = self.layer2(x)  # Output: (B, 32, 16, 16)

        # Stage 3
        f3 = self.layer3(f2)  # Output: (B, 64, 8, 8)

        # Multi-Scale Global Covariance Pooling
        # Extract second-order stats from mid-level and high-level features
        p2 = self.pool2(f2)  # Size: B x 528
        p3 = self.pool3(f3)  # Size: B x 2080

        # Feature Fusion
        out = torch.cat([p2, p3], dim=1)  # Size: B x 2608

        # Classification
        out = self.fc(out)

        return out
