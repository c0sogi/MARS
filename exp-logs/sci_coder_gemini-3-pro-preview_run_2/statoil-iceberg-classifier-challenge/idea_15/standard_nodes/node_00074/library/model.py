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


class DPCNetBlock(nn.Module):
    """
    A single block of the DPC-Net (Dual-Pooling Contracted Network).
    Structure: Conv -> BN -> ReLU -> CBAM -> DualPooling
    """

    def __init__(self, in_channels, out_channels):
        super(DPCNetBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.pool = DualPooling()  # Cite solution_lesson_node_00070

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.pool(x)
        return x


class DPCNet(nn.Module):
    def __init__(self):
        super(DPCNet, self).__init__()

        # --- Visual Branch (Backbone) ---
        # 4-Stage CNN with Dual Pooling
        # Backbone width strategy: [64, 128, 128, 64] (Cite solution_lesson_node_00049)

        # Block 1: Input 3 -> Conv 64 -> Pool (64+64=128)
        # 75x75 -> 37x37
        self.block1 = DPCNetBlock(config.NUM_INPUT_CHANNELS, 64)

        # Block 2: Input 128 -> Conv 128 -> Pool (128+128=256)
        # 37x37 -> 18x18
        self.block2 = DPCNetBlock(128, 128)

        # Block 3: Input 256 -> Conv 128 -> Pool (128+128=256)
        # 18x18 -> 9x9
        self.block3 = DPCNetBlock(256, 128)

        # Block 4: Input 256 -> Conv 64 (Contracted) -> Pool (64+64=128)
        # 9x9 -> 4x4
        # Cite solution_lesson_node_00041 (Channel Contraction)
        self.block4 = DPCNetBlock(256, config.CONTRACTED_FILTERS)

        # --- Classification Head ---
        # Final visual map is 128 x 4 x 4 = 2048
        # We concatenate the raw incidence angle (1 dim)
        # Cite solution_lesson_node_00073 (Concatenation over Gating)
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
        x = self.block4(x)  # Output: [Batch, 128, 4, 4]

        # Flatten visual features
        x = x.view(x.size(0), -1)  # Output: [Batch, 2048]

        # 2. Multimodal Fusion (Concatenation)
        # Cite solution_lesson_node_00073
        x = torch.cat([x, inc_angle], dim=1)  # Output: [Batch, 2049]

        # 3. Classification
        out = self.classifier(x)

        return out
