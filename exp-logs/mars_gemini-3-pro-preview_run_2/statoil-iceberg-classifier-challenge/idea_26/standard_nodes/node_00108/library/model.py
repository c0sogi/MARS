import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Uses Mixed Pooling (Avg + Max) followed by a shared MLP.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        # Reduction ratio for parameter efficiency
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Uses channel-wise pooling (Avg + Max) followed by a 7x7 convolution.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention and Spatial Attention.
    """

    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DualPoolingBlock(nn.Module):
    """
    Dual-Pooling Block without immediate integration.
    Conv -> BN -> ReLU -> CBAM -> DualPool.
    Output channels = 2 * out_channels.
    Cite solution_lesson_node_00107: Defer feature integration.
    """

    def __init__(self, in_channels, out_channels):
        super(DualPoolingBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cbam(x)

        # Dual-Stream Pooling (Cite solution_lesson_node_00070)
        x_max = self.max_pool(x)
        x_min = -self.max_pool(-x)

        return torch.cat([x_max, x_min], dim=1)


class DPCNet(nn.Module):
    """
    Dual-Pooling Contracted Network.
    Uses DualPoolingBlocks with delayed integration and channel contraction.
    """

    def __init__(self):
        super(DPCNet, self).__init__()

        # Block 1: 3 -> 64. Out: 128 (64*2)
        self.block1 = DualPoolingBlock(config.IN_CHANNELS, 64)
        # Block 2: 128 -> 128. Out: 256 (128*2)
        self.block2 = DualPoolingBlock(128, 128)
        # Block 3: 256 -> 128. Out: 256 (128*2)
        self.block3 = DualPoolingBlock(256, 128)
        # Block 4: 256 -> 32. Out: 64 (32*2) -> Channel Contraction (Cite solution_lesson_node_00041)
        self.block4 = DualPoolingBlock(256, 32)

        # Flattened size: 64 * 4 * 4 = 1024
        # Cite solution_lesson_node_00021: Flatten to retain coarse spatial geometry
        visual_dim = 1024

        # Metadata
        self.use_inc_angle = config.USE_INC_ANGLE
        if self.use_inc_angle:
            self.meta_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            fusion_dim = visual_dim + 32
        else:
            fusion_dim = visual_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_meta=None):
        x = self.block1(x_img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Flatten
        visual_vec = x.view(x.size(0), -1)

        # Metadata Branch
        if self.use_inc_angle and x_meta is not None:
            if x_meta.dim() == 1:
                x_meta = x_meta.unsqueeze(1)
            meta_vec = self.meta_mlp(x_meta)
            final_vec = torch.cat([visual_vec, meta_vec], dim=1)
        else:
            final_vec = visual_vec

        logits = self.classifier(final_vec)
        return logits
