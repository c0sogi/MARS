import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CovariancePooling(nn.Module):
    """
    Implements Global Covariance Pooling with Newton-Schulz normalization.
    Captures second-order statistics (texture) from feature maps.
    """

    def __init__(self, num_features, num_iterations=3):
        super(CovariancePooling, self).__init__()
        self.num_features = num_features
        self.num_iterations = num_iterations
        # The output dimension is the number of unique elements in the upper triangle
        # of the covariance matrix: C * (C + 1) / 2
        self.output_dim = int(num_features * (num_features + 1) / 2)

    def _newton_schulz_sqrt(self, A, num_iters):
        """
        Computes the matrix square root using Newton-Schulz iteration.
        Y_{k+1} = 0.5 * Y_k * (3I - Z_k * Y_k)
        Z_{k+1} = 0.5 * (3I - Z_k * Y_k) * Z_k
        """
        batch_size = A.shape[0]
        dim = A.shape[1]

        # Normalization by trace to ensure convergence
        # A_norm = A / tr(A)
        trace = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        # Add epsilon to avoid division by zero
        trace = torch.clamp(trace, min=1e-6)
        A_norm = A / trace

        Y = A_norm
        Z = torch.eye(dim, device=A.device).unsqueeze(0).repeat(batch_size, 1, 1)
        I = torch.eye(dim, device=A.device).unsqueeze(0).repeat(batch_size, 1, 1)

        for _ in range(num_iters):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)

        # Denormalize: sqrt(A) = sqrt(trace) * Y_final
        sqrt_trace = torch.sqrt(trace)
        return Y * sqrt_trace

    def _get_upper_triangular(self, cov_matrix):
        """
        Extracts the upper triangular part of the covariance matrix and flattens it.
        """
        B, C, _ = cov_matrix.shape
        # Create indices for upper triangular part
        idx = torch.triu_indices(C, C, device=cov_matrix.device)
        # Gather elements: (B, num_upper)
        flat = cov_matrix[:, idx[0], idx[1]]
        return flat

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        N = H * W

        # Reshape to (B, C, N)
        x = x.view(B, C, N)

        # Center the features
        mean = x.mean(dim=2, keepdim=True)
        x_centered = x - mean

        # Compute Covariance Matrix: (1 / (N-1)) * X * X^T
        # Result shape: (B, C, C)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (N - 1 + 1e-5)

        # Apply Newton-Schulz Normalization (Matrix Square Root)
        # This acts as a power normalization to improve classifier performance
        cov_sqrt = self._newton_schulz_sqrt(cov, self.num_iterations)

        # Flatten upper triangular part
        out = self._get_upper_triangular(cov_sqrt)

        return out


class WideBasicBlock(nn.Module):
    """
    Standard Residual Block for Wide ResNet.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HybridCactusClassifier(nn.Module):
    """
    Custom Wide ResNet with Hybrid Texture-Semantic Aggregation.

    Backbone:
        - 3 Stages of WideBasicBlocks.
        - Channels: [32, 64, 128].
        - Downsampling: 32x32 -> 16x16 -> 8x8.

    Head:
        - Stream 1 (Texture): Stage 2 output (16x16, 64ch) -> Covariance Pooling.
        - Stream 2 (Semantic): Stage 3 output (8x8, 128ch) -> Global Average Pooling.
        - Fusion: Concatenation -> Linear.
    """

    def __init__(self):
        super(HybridCactusClassifier, self).__init__()

        # Configuration
        channels = Config.BACKBONE_CHANNELS  # [32, 64, 128]

        # Initial Convolution
        # Input: (B, 3, 32, 32) -> Output: (B, 32, 32, 32)
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                Config.INPUT_CHANNELS,
                channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 channels, 32x32 resolution
        # We use 2 blocks per stage to ensure capacity
        self.layer1 = self._make_layer(channels[0], channels[0], stride=1, num_blocks=2)

        # Stage 2: 64 channels, 16x16 resolution (Stride 2)
        self.layer2 = self._make_layer(channels[0], channels[1], stride=2, num_blocks=2)

        # Stage 3: 128 channels, 8x8 resolution (Stride 2)
        self.layer3 = self._make_layer(channels[1], channels[2], stride=2, num_blocks=2)

        # --- Hybrid Head ---

        # Stream 1: Texture (Covariance Pooling)
        # Applied to Stage 2 output (64 channels)
        self.texture_pool = CovariancePooling(channels[1])
        self.texture_dim = self.texture_pool.output_dim  # 64*(65)/2 = 2080

        # Stream 2: Semantic (Global Average Pooling)
        # Applied to Stage 3 output (128 channels)
        self.semantic_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.semantic_dim = channels[2]  # 128

        # Classifier
        total_dim = self.texture_dim + self.semantic_dim
        self.fc = nn.Linear(total_dim, Config.NUM_CLASSES)

        # Initialization
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(WideBasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

    def forward(self, x):
        # Initial Conv
        x = self.init_conv(x)

        # Stage 1
        x1 = self.layer1(x)

        # Stage 2 (Texture Source)
        x2 = self.layer2(x1)

        # Stage 3 (Semantic Source)
        x3 = self.layer3(x2)

        # --- Hybrid Aggregation ---

        # Stream 1: Texture from Stage 2
        # x2 shape: (B, 64, 16, 16)
        texture_feat = self.texture_pool(x2)

        # Stream 2: Semantic from Stage 3
        # x3 shape: (B, 128, 8, 8)
        semantic_feat = self.semantic_pool(x3)
        semantic_feat = torch.flatten(semantic_feat, 1)

        # Fusion
        combined_feat = torch.cat([texture_feat, semantic_feat], dim=1)

        # Classification
        logits = self.fc(combined_feat)

        # Squeeze to match target shape (B,) if necessary, but usually (B, 1) is fine for BCEWithLogits
        # However, utils.calculate_roc_auc expects flat arrays usually.
        # We return (B, 1) logits.
        return logits
