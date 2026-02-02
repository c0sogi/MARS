import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization.
    Predicts affine parameters (gamma, beta) from the incidence angle using a lightweight MLP.
    """

    def __init__(self, num_features, hidden_dim=16):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        # Standard BN to compute mean/var, but without learnable affine params
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # MLP to map scalar angle to gamma and beta
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_features),
        )

        # Initialize the last linear layer to zeros.
        # This ensures gamma starts at 0 (effectively 1 in our formula) and beta at 0.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, angle):
        """
        Args:
            x: Feature map of shape (B, C, H, W)
            angle: Incidence angles of shape (B,) or (B, 1)
        """
        # Normalize x using running statistics
        out = self.bn(x)

        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Predict affine parameters
        params = self.mlp(angle)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # (B, C) each

        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)

        # Apply affine transformation: y = x * (1 + gamma) + beta
        return out * (1.0 + gamma) + beta


class ConditionalBasicBlock(nn.Module):
    """
    Residual Block using Conditional Batch Normalization.
    Passes 'angle' to normalization layers.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConditionalBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.cbn1 = ConditionalBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.cbn2 = ConditionalBatchNorm2d(out_channels)

        self.downsample_conv = None
        self.downsample_cbn = None

        # If stride > 1 or channels change, we need a downsampling path on the identity
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.downsample_cbn = ConditionalBatchNorm2d(out_channels)

    def forward(self, x, angle):
        identity = x

        out = self.conv1(x)
        out = self.cbn1(out, angle)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.cbn2(out, angle)

        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_cbn(identity, angle)

        out += identity
        out = self.relu(out)

        return out


class ACResNet(nn.Module):
    """
    Angle-Calibrated Residual Network (AC-ResNet).
    A shallow ResNet architecture that conditions feature normalization on the radar incidence angle.
    """

    def __init__(self):
        super(ACResNet, self).__init__()

        # Hyperparameters from Config
        widths = Config.CHANNEL_WIDTHS  # Expected: [64, 64, 128]
        fc_dim = Config.FC_DIM
        dropout_rate = Config.DROPOUT_RATE
        in_channels = Config.IN_CHANNELS

        # --- Stem ---
        self.conv1 = nn.Conv2d(
            in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.cbn1 = ConditionalBatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)

        # --- Stages ---
        # Stage 1: 64 channels
        self.stage1 = self._make_stage(widths[0], widths[0], stride=1, num_blocks=2)
        # Stage 2: 64 channels
        self.stage2 = self._make_stage(widths[0], widths[1], stride=2, num_blocks=2)
        # Stage 3: 128 channels
        self.stage3 = self._make_stage(widths[1], widths[2], stride=2, num_blocks=2)

        # --- Head ---
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        self.fc_block = nn.Sequential(
            nn.Linear(widths[2], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _make_stage(self, in_channels, out_channels, stride, num_blocks):
        """
        Creates a stage of residual blocks.
        Returns a ModuleList to allow custom forward pass with 'angle'.
        """
        layers = []
        # The first block handles the stride and channel transition
        layers.append(ConditionalBasicBlock(in_channels, out_channels, stride=stride))
        # Subsequent blocks preserve dimensions
        for _ in range(1, num_blocks):
            layers.append(ConditionalBasicBlock(out_channels, out_channels, stride=1))
        return nn.ModuleList(layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # BN weights are handled in ConditionalBatchNorm2d initialization

    def forward(self, x, angle):
        """
        Forward pass.
        Args:
            x: Image tensor (B, 3, 75, 75)
            angle: Incidence angle tensor (B,)
        Returns:
            Logits (B, 1)
        """
        # Stem
        x = self.conv1(x)
        x = self.cbn1(x, angle)
        x = self.relu(x)

        # Stage 1
        for block in self.stage1:
            x = block(x, angle)

        # Stage 2
        for block in self.stage2:
            x = block(x, angle)

        # Stage 3
        for block in self.stage3:
            x = block(x, angle)

        # Head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)

        return x
