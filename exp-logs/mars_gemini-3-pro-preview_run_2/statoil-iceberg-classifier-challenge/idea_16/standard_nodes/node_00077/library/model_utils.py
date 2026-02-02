import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) from CBAM.
    Aggregates spatial information using both avg and max pooling.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Use a reduction ratio to limit parameter count
        # For small channel counts (e.g. 64), ratio=16 gives bottleneck of 4.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

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
    Spatial Attention Module (SAM) from CBAM.
    Focuses on 'where' the informative part is.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # Input is 2 channels: 1 from max pool, 1 from avg pool
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel and Spatial attention.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class DualPooling(nn.Module):
    """
    Dual-Stream Pooling Module.
    Computes both Max and Min pooling and concatenates them.
    Cite solution_lesson_node_00070: Preserving Signal Extremes.
    """

    def __init__(self, kernel_size=2, stride=2):
        super(DualPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        # Max Pooling (Peaks)
        out_max = self.max_pool(x)
        # Min Pooling (Shadows) - implemented as negative max of negative input
        out_min = -self.max_pool(-x)
        return torch.cat([out_max, out_min], dim=1)


class CSNet(nn.Module):
    """
    Dual-Pooling Contracted Network (DPCNet).
    Replaces input-level augmentation with architectural Dual Pooling.

    Features:
    1. Dual Pooling: Preserves both shadows and peaks (Cite Lesson 70, 76).
    2. Contracted Architecture: 32 -> 64 -> 64 -> 32 filters (Cite Lesson 41).
    3. CBAM Attention: Applied Pre-Pooling (Cite Lesson 61).
    """

    def __init__(self):
        super(CSNet, self).__init__()

        # --- Visual Branch ---
        # Input: 3 channels (HH, HV, Avg)

        # Stage 1: 3 -> 32 (Conv) -> 32 (CBAM) -> 64 (DualPool)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        # Stage 2: 64 -> 64 (Conv) -> 64 (CBAM) -> 128 (DualPool)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)

        # Stage 3: 128 -> 64 (Conv) -> 64 (CBAM) -> 128 (DualPool)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.cbam3 = CBAM(64)

        # Stage 4: 128 -> 32 (Conv) -> 32 (CBAM) -> 64 (DualPool)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.cbam4 = CBAM(32)

        self.pool = DualPooling(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # --- Metadata Branch ---
        # Processes scalar incidence angle
        self.meta_fc1 = nn.Linear(1, 16)
        self.meta_bn1 = nn.BatchNorm1d(16)
        self.meta_fc2 = nn.Linear(16, 32)
        self.meta_bn2 = nn.BatchNorm1d(32)

        # --- Fusion Head ---
        # Visual flat: 64 (channels) * 4 * 4 = 1024
        # Meta flat: 32
        # Total: 1056
        self.fc1 = nn.Linear(1024 + 32, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, 1)

        # Weight initialization
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

    def forward(self, x_img, x_angle):
        # --- Visual Branch Forward ---
        # Stage 1
        x = self.conv1(x_img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool(x)  # Output channels doubles here

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cbam2(x)
        x = self.pool(x)

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.cbam3(x)
        x = self.pool(x)

        # Stage 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.cbam4(x)
        x = self.pool(x)

        # Flatten: (N, 64, 4, 4) -> (N, 1024)
        x_visual = x.view(x.size(0), -1)

        # --- Metadata Branch Forward ---
        # Ensure angle is (N, 1)
        if len(x_angle.shape) == 1:
            x_angle = x_angle.unsqueeze(1)

        m = self.meta_fc1(x_angle)
        m = self.meta_bn1(m)
        m = self.relu(m)
        m = self.meta_fc2(m)
        m = self.meta_bn2(m)
        m = self.relu(m)  # (N, 32)

        # --- Fusion ---
        combined = torch.cat((x_visual, m), dim=1)

        out = self.fc1(combined)
        out = self.bn_fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.out(out)  # Returns Logits

        return out
