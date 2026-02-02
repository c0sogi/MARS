import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import INPUT_SHAPE, CHANNEL_CONFIG, NUM_CLASSES


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block to adaptively recalibrate channel-wise feature responses.
    """

    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block with SE module.
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        self.se = SEBlock(planes, reduction=8)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GlobalCovariancePooling(nn.Module):
    """
    Global Covariance Pooling with Newton-Schulz Matrix Square Root Normalization.
    Captures second-order statistics (texture) from feature maps.
    """

    def __init__(self, num_features):
        super(GlobalCovariancePooling, self).__init__()
        self.num_features = num_features
        # The dimension of the flattened upper triangular matrix
        self.out_dim = num_features * (num_features + 1) // 2

    def _newton_schulz_sqrt(self, A, num_iters=3):
        """
        Computes the matrix square root using Newton-Schulz iterations.
        """
        batch_size = A.shape[0]
        dim = A.shape[1]

        # Normalize by trace to ensure convergence
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batch_size, 1, 1).expand_as(A))
        I = (
            torch.eye(dim, dim, device=A.device)
            .view(1, dim, dim)
            .repeat(batch_size, 1, 1)
        )
        Z = (
            torch.eye(dim, dim, device=A.device)
            .view(1, dim, dim)
            .repeat(batch_size, 1, 1)
        )

        for i in range(num_iters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # Rescale
        sA = Y * normA.view(batch_size, 1, 1).expand_as(A).sqrt()
        return sA

    def _triu_flatten(self, A):
        """
        Extracts and flattens the upper triangular part of the matrix.
        """
        b, c, _ = A.shape
        idx = torch.triu_indices(c, c, device=A.device)
        # Result shape: (B, C*(C+1)/2)
        return A[:, idx[0], idx[1]]

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, C, N)
        features = x.view(B, C, N)

        # Center features (subtract mean)
        mean = features.mean(dim=2, keepdim=True)
        features = features - mean

        # Compute Covariance Matrix: (1/N) * X * X^T
        cov = torch.bmm(features, features.transpose(1, 2)).div(N)

        # Add small epsilon to diagonal for numerical stability
        cov = cov + 1e-5 * torch.eye(C, device=x.device).unsqueeze(0)

        # Apply Matrix Square Root Normalization
        sqrt_cov = self._newton_schulz_sqrt(cov)

        # Flatten Upper Triangular part
        out = self._triu_flatten(sqrt_cov)

        return out


class NarrowSEResNet(nn.Module):
    """
    Custom Narrow SE-ResNet with Selective Texture-Context Aggregation.
    """

    def __init__(self):
        super(NarrowSEResNet, self).__init__()

        # Channel Configuration: [16, 32, 64]
        c1, c2, c3 = CHANNEL_CONFIG

        # Stem: 32x32 input -> 32x32 feature map
        self.conv1 = nn.Conv2d(
            INPUT_SHAPE[0], c1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(c1)

        # Stage 1: 16 channels, 32x32 resolution
        self.layer1 = self._make_layer(c1, c1, stride=1)

        # Stage 2: 32 channels, 16x16 resolution
        self.layer2 = self._make_layer(c1, c2, stride=2)

        # Stage 3: 64 channels, 8x8 resolution
        self.layer3 = self._make_layer(c2, c3, stride=2)

        # --- Selective Aggregation Heads ---

        # Texture Stream: Operates on Stage 2 (16x16, 32 ch)
        # N=256 >> C=32, suitable for Covariance Pooling
        self.gcp = GlobalCovariancePooling(c2)

        # Context Stream: Operates on Stage 3 (8x8, 64 ch)
        # N=64 approx C=64, suitable for Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        # GCP output dim: 32 * 33 / 2 = 528
        # GAP output dim: 64
        self.fc_input_dim = self.gcp.out_dim + c3
        self.fc = nn.Linear(self.fc_input_dim, NUM_CLASSES)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))

        # Stage 1
        out = self.layer1(out)

        # Stage 2 (Texture Source)
        stage2_out = self.layer2(out)  # Shape: (B, 32, 16, 16)

        # Stage 3 (Context Source)
        stage3_out = self.layer3(stage2_out)  # Shape: (B, 64, 8, 8)

        # --- Aggregation ---

        # Texture Stream: Global Covariance Pooling on Stage 2
        texture_feat = self.gcp(stage2_out)  # Shape: (B, 528)

        # Context Stream: Global Average Pooling on Stage 3
        context_feat = self.gap(stage3_out)  # Shape: (B, 64, 1, 1)
        context_feat = context_feat.view(context_feat.size(0), -1)  # Shape: (B, 64)

        # Fusion
        fused_feat = torch.cat([texture_feat, context_feat], dim=1)  # Shape: (B, 592)

        # Classification
        logits = self.fc(fused_feat)

        return logits
