import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure hidden dimension is at least 1
        hidden_dim = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GlobalCovariancePooling(nn.Module):
    """
    Global Covariance Pooling with Newton-Schulz Matrix Square Root Normalization.
    Captures second-order statistics (texture/correlation) of the features.
    """

    def __init__(self, input_dim):
        super(GlobalCovariancePooling, self).__init__()
        self.input_dim = input_dim
        # Number of unique elements in the upper triangle of a DxD matrix
        self.output_dim = int(input_dim * (input_dim + 1) / 2)

    def _newton_schulz_sqrt(self, A, num_iters=3):
        """
        Iterative method to compute the matrix square root.
        """
        batch_size = A.shape[0]
        dim = A.shape[1]

        # Normalize A to ensure convergence
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batch_size, 1, 1).expand_as(A) + 1e-6)

        I = torch.eye(dim, device=A.device).view(1, dim, dim).expand_as(A)
        Z = torch.eye(dim, device=A.device).view(1, dim, dim).expand_as(A)

        for _ in range(num_iters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)

        # Denormalize
        return Y.mul(normA.sqrt().view(batch_size, 1, 1).expand_as(A))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, C, N)
        features = x.view(B, C, N)

        # Center features (subtract mean)
        mean = features.mean(dim=2, keepdim=True)
        features = features - mean

        # Compute Covariance Matrix: (B, C, C)
        # 1/(N-1) * X * X^T
        # Add epsilon for stability
        cov = torch.bmm(features, features.transpose(1, 2)).div(
            max(1, N - 1)
        ) + 1e-5 * torch.eye(C, device=x.device).unsqueeze(0)

        # Apply Matrix Square Root Normalization
        sqrt_cov = self._newton_schulz_sqrt(cov)

        # Flatten Upper Triangular part
        indices = torch.triu_indices(C, C, device=x.device)
        # Gather values: (B, num_features)
        out = sqrt_cov[:, indices[0], indices[1]]

        return out


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block with optional SE Module.
    """

    def __init__(self, in_planes, planes, stride=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = self.se(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HybridNarrowSEResNet(nn.Module):
    """
    Custom Narrow SE-ResNet with Hybrid Multi-Order Multi-Scale Pooling.
    Combines GAP (First-order) and GCP (Second-order) features from multiple resolution stages.
    """

    def __init__(self):
        super(HybridNarrowSEResNet, self).__init__()

        channels = Config.BACKBONE_CHANNELS  # Expected: [16, 32, 64]
        use_se = Config.USE_SE

        self.in_planes = channels[0]

        # --- Backbone ---
        # Initial Stem
        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        # Stages
        # Stage 1: 32x32 -> 32x32 (Stride 1)
        self.layer1 = self._make_layer(channels[0], stride=1, use_se=use_se)
        # Stage 2: 32x32 -> 16x16 (Stride 2)
        self.layer2 = self._make_layer(channels[1], stride=2, use_se=use_se)
        # Stage 3: 16x16 -> 8x8 (Stride 2)
        self.layer3 = self._make_layer(channels[2], stride=2, use_se=use_se)

        # --- Hybrid Pooling Head ---
        self.pool_indices = Config.POOLING_STAGES_INDICES

        # Define GCP layers for the selected stages
        self.gcp_layers = nn.ModuleDict()

        total_features = 0

        for idx in self.pool_indices:
            ch = channels[idx]

            # 1. GAP contribution: ch features
            gap_dim = ch

            # 2. GCP contribution: ch * (ch + 1) / 2 features
            gcp_layer = GlobalCovariancePooling(ch)
            gcp_dim = gcp_layer.output_dim

            self.gcp_layers[str(idx)] = gcp_layer

            total_features += gap_dim + gcp_dim

        # Classifier
        self.fc = nn.Linear(total_features, Config.NUM_CLASSES)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, stride, use_se):
        # Using 2 blocks per stage for robustness
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, use_se))
        self.in_planes = planes
        layers.append(BasicBlock(self.in_planes, planes, 1, use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))

        # Backbone Forward Pass
        feats = {}

        # Stage 1 (Index 0)
        x = self.layer1(x)
        feats[0] = x

        # Stage 2 (Index 1)
        x = self.layer2(x)
        feats[1] = x

        # Stage 3 (Index 2)
        x = self.layer3(x)
        feats[2] = x

        # Hybrid Pooling & Fusion
        pooled_features = []

        for idx in self.pool_indices:
            f_map = feats[idx]

            # 1. Global Average Pooling (First-Order)
            # (B, C, H, W) -> (B, C)
            gap = torch.mean(f_map, dim=[2, 3])
            pooled_features.append(gap)

            # 2. Global Covariance Pooling (Second-Order)
            # (B, C, H, W) -> (B, C*(C+1)/2)
            gcp = self.gcp_layers[str(idx)](f_map)
            pooled_features.append(gcp)

        # Concatenate all features
        final_vec = torch.cat(pooled_features, dim=1)

        # Classification
        out = self.fc(final_vec)

        return out
