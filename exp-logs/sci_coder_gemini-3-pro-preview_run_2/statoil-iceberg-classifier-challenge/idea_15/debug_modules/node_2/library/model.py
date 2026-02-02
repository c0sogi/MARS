import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Ensure hidden dimension is at least 1
        hidden_planes = max(1, in_planes // ratio)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling: Concatenates Max Pooling and Min Pooling outputs.
    Doubles the number of channels.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Max Pooling
        max_p = F.max_pool2d(x, self.kernel_size, self.stride)
        # Min Pooling: -MaxPool(-x)
        min_p = -F.max_pool2d(-x, self.kernel_size, self.stride)
        # Concatenate along channel dimension
        return torch.cat([max_p, min_p], dim=1)


class GDPNetBlock(nn.Module):
    """
    A single block of the GDP-Net Visual Branch.
    Structure: Conv -> BN -> ReLU -> CBAM -> DualPooling
    """

    def __init__(self, in_channels, out_channels):
        super(GDPNetBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.pool = DualPooling()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x


class GDPNet(nn.Module):
    def __init__(self):
        super(GDPNet, self).__init__()

        # --- Visual Branch (Backbone) ---
        # 4-Stage CNN with Dual Pooling

        # Block 1: Input 3 -> Conv 32 -> Pool (32+32=64)
        # 75x75 -> 37x37
        self.block1 = GDPNetBlock(config.NUM_INPUT_CHANNELS, 32)

        # Block 2: Input 64 -> Conv 48 -> Pool (48+48=96)
        # 37x37 -> 18x18
        self.block2 = GDPNetBlock(64, 48)

        # Block 3: Input 96 -> Conv 64 -> Pool (64+64=128)
        # 18x18 -> 9x9
        self.block3 = GDPNetBlock(96, 64)

        # Block 4: Input 128 -> Conv 32 (Contracted) -> Pool (32+32=64)
        # 9x9 -> 4x4
        self.block4 = GDPNetBlock(128, config.CONTRACTED_FILTERS)

        # --- Metadata Branch (Gating Generator) ---
        # Maps incidence angle to a gating vector matching the visual feature depth (64)
        self.meta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, config.GATING_VECTOR_DIM),
            nn.Sigmoid(),
        )

        # --- Classification Head ---
        # Final visual map is 64 x 4 x 4 = 1024
        # We concatenate the raw incidence angle (1 dim)
        self.head_input_dim = (config.CONTRACTED_FILTERS * 2 * 4 * 4) + 1

        self.classifier = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, inc_angle):
        # x: [Batch, 3, 75, 75]
        # inc_angle: [Batch] or [Batch, 1]

        # Ensure inc_angle is [Batch, 1]
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.view(-1, 1)

        # 1. Visual Branch
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Output: [Batch, 64, 4, 4]

        # 2. Metadata Branch (Gating)
        gating_vector = self.meta_mlp(inc_angle)  # Output: [Batch, 64]

        # 3. Gated Fusion
        # Reshape gate for broadcasting: [Batch, 64, 1, 1]
        gate = gating_vector.view(-1, config.GATING_VECTOR_DIM, 1, 1)

        # Element-wise multiplication (Modulation)
        x = x * gate

        # Flatten visual features
        x = x.view(x.size(0), -1)  # Output: [Batch, 1024]

        # Concatenate with raw incidence angle
        x = torch.cat([x, inc_angle], dim=1)  # Output: [Batch, 1025]

        # 4. Classification
        out = self.classifier(x)

        return out
