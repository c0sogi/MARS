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


class ExpandContractBlock(nn.Module):
    """
    Implements the "Expand-Then-Contract" topology:
    1. Wide Conv (128)
    2. CBAM Attention (Pre-Pooling)
    3. Dual-Stream Pooling (Max + Min) -> Expands to 256
    4. 1x1 Contraction -> Contracts to 128
    """

    def __init__(self, in_channels, out_channels):
        super(ExpandContractBlock, self).__init__()

        # 1. Wide Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Attention
        self.cbam = CBAM(out_channels)

        # 3. Dual-Stream Pooling is functional (Max and Min), no learnable params here
        # We will define the contraction layer based on the concatenated size

        # 4. Immediate Contraction
        # Input to this will be out_channels * 2 (Max stream + Min stream)
        self.contraction = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1, bias=False
        )
        self.bn_contract = nn.BatchNorm2d(out_channels)
        self.relu_contract = nn.ReLU(inplace=True)

    def forward(self, x):
        # Wide Conv
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Attention (Pre-Pooling)
        x = self.cbam(x)

        # Dual-Stream Pooling
        # Max Pooling (Peaks)
        x_max = F.max_pool2d(x, kernel_size=2, stride=2)

        # Min Pooling (Shadows)
        # Simulated by inverting input, max pooling, and inverting back
        x_min = -F.max_pool2d(-x, kernel_size=2, stride=2)

        # Concatenate: (B, 2*C, H/2, W/2)
        x_pool = torch.cat([x_max, x_min], dim=1)

        # Immediate Contraction
        x_out = self.contraction(x_pool)
        x_out = self.bn_contract(x_out)
        x_out = self.relu_contract(x_out)

        return x_out


class QuadrantMaxPooling(nn.Module):
    """
    Adaptive Max Pooling to a 2x2 grid.
    Preserves spatial quadrant information while reducing dimensionality.
    """

    def __init__(self):
        super(QuadrantMaxPooling, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((2, 2))

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, 2, 2)
        x = self.pool(x)
        # Flatten: (B, C * 2 * 2)
        x = x.view(x.size(0), -1)
        return x


class PPCWBN(nn.Module):
    """
    Post-Pooling Contracted Wide-Body Network.
    """

    def __init__(self):
        super(PPCWBN, self).__init__()

        # Configuration
        self.width = Config.BACKBONE_WIDTH  # 128
        self.dropout_rate = Config.DROPOUT_RATE  # 0.5

        # --- Visual Branch ---
        # Input: 3 channels (Band1, Band2, Mean)
        # Stage 1: 75 -> 37
        self.stage1 = ExpandContractBlock(Config.IN_CHANNELS, self.width)
        # Stage 2: 37 -> 18
        self.stage2 = ExpandContractBlock(self.width, self.width)
        # Stage 3: 18 -> 9
        self.stage3 = ExpandContractBlock(self.width, self.width)
        # Stage 4: 9 -> 4
        self.stage4 = ExpandContractBlock(self.width, self.width)

        # Readout
        self.quadrant_pool = QuadrantMaxPooling()
        # Visual vector size: 128 channels * 2 * 2 = 512
        self.visual_dim = self.width * 4

        # --- Metadata Branch ---
        # Simple MLP for incidence angle
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
        self.classifier = nn.Linear(256, 1)  # Binary classification logits

        # Weight Initialization
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

        # Quadrant Pooling -> Flatten
        v_feat = self.quadrant_pool(x)

        # --- Metadata Branch ---
        # Ensure angle is (B, 1)
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
