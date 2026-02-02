import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        # Ensure reduction doesn't make hidden channels too small
        mid_channels = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=True),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Linear(mid_channels, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """
    Residual Block with Stochastic Depth and SE Attention.
    Uses LeakyReLU and retains Bias.
    """

    def __init__(self, in_channels, out_channels, stride=1, drop_path=0.0):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=Config.USE_BIAS,
        )
        self.act1 = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=Config.USE_BIAS,
        )

        self.se = SEModule(out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.act2 = nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True)

        # Shortcut handling
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=Config.USE_BIAS,
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.se(out)

        out = self.drop_path(out)
        out += residual
        out = self.act2(out)

        return out


class SDHAResNet(nn.Module):
    """
    Stochastic-Depth Hybrid-Attentive ResNet (SDHA-ResNet).

    Architecture:
    - Stem
    - 4 Stages of ResBlocks with increasing DropPath rates
    - Global Max Pooling
    - Fusion with raw incidence angle
    - Classification Head
    """

    def __init__(self):
        super(SDHAResNet, self).__init__()

        # Configuration
        in_ch = Config.IN_CHANNELS
        widths = Config.CHANNEL_WIDTHS  # [64, 128, 128, 128]
        # Assuming 2 blocks per stage for the "Custom 4-Stage" design
        layers = [2, 2, 2, 2]
        num_blocks = sum(layers)
        dpr = [x.item() for x in torch.linspace(0, Config.DROP_PATH_RATE, num_blocks)]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_ch,
                widths[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=Config.USE_BIAS,
            ),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
        )

        self.stages = nn.ModuleList()
        current_idx = 0
        in_planes = widths[0]

        # Build 4 Stages
        for i in range(Config.STAGES):
            out_planes = widths[i]
            stage_blocks = []

            # For each stage, we have 'layers[i]' blocks.
            # The first block of each stage (except potentially the first stage if we didn't want downsampling there)
            # handles stride. The prompt says "4 downsampling operations".
            # Input is 75x75.
            # Stage 1: Stride 2 -> 38x38
            # Stage 2: Stride 2 -> 19x19
            # Stage 3: Stride 2 -> 10x10
            # Stage 4: Stride 2 -> 5x5
            # So every stage starts with stride 2.

            stride = 2

            # First block of the stage
            stage_blocks.append(
                ResBlock(
                    in_planes, out_planes, stride=stride, drop_path=dpr[current_idx]
                )
            )
            current_idx += 1
            in_planes = out_planes

            # Subsequent blocks in the stage
            for _ in range(1, layers[i]):
                stage_blocks.append(
                    ResBlock(
                        in_planes, out_planes, stride=1, drop_path=dpr[current_idx]
                    )
                )
                current_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

        # Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Head
        # Input features + 1 scalar (angle)
        head_in_dim = widths[-1] + 1
        hidden_dim = 128

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.LeakyReLU(Config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(Config.HEAD_DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # x: (B, 3, 75, 75)
        # angle: (B,)

        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Global Max Pooling
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Feature Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        x = torch.cat([x, angle], dim=1)

        # Classification
        logits = self.head(x)
        return logits.squeeze(1)
