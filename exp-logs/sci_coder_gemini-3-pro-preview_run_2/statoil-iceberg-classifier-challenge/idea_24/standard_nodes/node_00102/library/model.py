import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
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
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class ContractedDualPoolBlock(nn.Module):
    """
    Implements the "Contract-Then-Expand" topology (Cite solution_lesson_node_00092):
    1. Wide Conv (out_channels)
    2. CBAM Attention (Pre-Pooling) (Cite solution_lesson_node_00061)
    3. 1x1 Contraction -> Contracts to out_channels // 2
    4. Dual-Stream Pooling (Max + Min) -> Expands back to out_channels (Cite solution_lesson_node_00070)
    """

    def __init__(self, in_channels, out_channels):
        super(ContractedDualPoolBlock, self).__init__()

        # 1. Wide Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Attention (Pre-Pooling)
        self.cbam = CBAM(out_channels)

        # 3. Contraction (Bottleneck)
        # We contract to half the desired output size, because DualPooling will double it.
        self.bottleneck = nn.Conv2d(
            out_channels, out_channels // 2, kernel_size=1, bias=False
        )
        self.bn_bot = nn.BatchNorm2d(out_channels // 2)
        self.relu_bot = nn.ReLU(inplace=True)

    def forward(self, x):
        # Wide Conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Attention
        x = self.cbam(x)

        # Contraction
        x = self.bottleneck(x)
        x = self.bn_bot(x)
        x = self.relu_bot(x)

        # Dual-Stream Pooling (Max + Min)
        x_max = F.max_pool2d(x, kernel_size=2, stride=2)
        x_min = -F.max_pool2d(-x, kernel_size=2, stride=2)

        # Concatenate: (B, C/2 * 2, H/2, W/2) -> (B, C, H/2, W/2)
        return torch.cat([x_max, x_min], dim=1)


class DPCNet(nn.Module):
    """
    Dual-Pooling Contracted Network.
    """

    def __init__(self):
        super(DPCNet, self).__init__()

        self.dropout_rate = Config.DROPOUT_RATE

        # --- Visual Branch ---
        # Input: 3 channels
        # Stage 1: 3 -> 64
        self.stage1 = ContractedDualPoolBlock(Config.IN_CHANNELS, 64)
        # Stage 2: 64 -> 128
        self.stage2 = ContractedDualPoolBlock(64, 128)
        # Stage 3: 128 -> 128
        self.stage3 = ContractedDualPoolBlock(128, 128)
        # Stage 4: 128 -> 64 (Structural Contraction) (Cite solution_lesson_node_00041)
        self.stage4 = ContractedDualPoolBlock(128, 64)

        # Readout: Flatten
        # Output of Stage 4 is 64 channels. 75 -> 37 -> 18 -> 9 -> 4.
        # 64 * 4 * 4 = 1024
        self.visual_dim = 64 * 4 * 4

        # --- Metadata Branch ---
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_bn1 = nn.BatchNorm1d(16)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_bn2 = nn.BatchNorm1d(32)
        self.meta_dim = 32

        # --- Fusion Head ---
        fusion_dim = self.visual_dim + self.meta_dim

        self.head_fc = nn.Linear(fusion_dim, 256)
        self.head_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(256, 1)

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

    def forward(self, x_img, x_angle):
        # --- Visual Branch ---
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Flatten
        v_feat = x.view(x.size(0), -1)

        # --- Metadata Branch ---
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)

        m_feat = self.meta_fc1(x_angle)
        m_feat = self.meta_bn1(m_feat)
        m_feat = F.relu(m_feat)

        m_feat = self.meta_fc2(m_feat)
        m_feat = self.meta_bn2(m_feat)
        m_feat = F.relu(m_feat)

        # --- Fusion ---
        combined = torch.cat([v_feat, m_feat], dim=1)

        # Head
        out = self.head_fc(combined)
        out = self.head_bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.classifier(out)

        return out
