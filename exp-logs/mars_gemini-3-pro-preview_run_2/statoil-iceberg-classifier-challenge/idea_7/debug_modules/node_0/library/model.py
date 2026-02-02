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

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Takes 2 channels (max pool + avg pool) -> 1 channel output
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return torch.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel Attention refinement
        out = x * self.ca(x)
        # Spatial Attention refinement
        out = out * self.sa(out)
        return out


class VisualBranch(nn.Module):
    def __init__(self):
        super(VisualBranch, self).__init__()

        # Input: (Batch, 3, 75, 75)

        # Block 1: 75x75 -> 37x37
        self.layer1 = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 37x37 -> 18x18
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAMBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 18x18 -> 9x9
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            CBAMBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 9x9 -> 4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Output dimension calculation: 64 channels * 4 * 4 spatial
        self.output_dim = 64 * 4 * 4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Flatten
        x = x.view(x.size(0), -1)
        return x


class StatisticalBranch(nn.Module):
    def __init__(self):
        super(StatisticalBranch, self).__init__()

        self.output_dim = 64

        self.mlp = nn.Sequential(
            nn.Linear(Config.NUM_STAT_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class SEAHN(nn.Module):
    """
    Statistically-Enriched Attention Hybrid Network (SEA-HN)
    Combines a CBAM-enhanced CNN for spatial features with an MLP for statistical features.
    """

    def __init__(self):
        super(SEAHN, self).__init__()

        self.visual_branch = VisualBranch()
        self.stat_branch = StatisticalBranch()

        # Fusion Head
        fusion_dim = self.visual_branch.output_dim + self.stat_branch.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 1),
        )

    def forward(self, img, stats):
        # Process Visual Branch
        v_out = self.visual_branch(img)

        # Process Statistical Branch
        s_out = self.stat_branch(stats)

        # Concatenate
        combined = torch.cat((v_out, s_out), dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits
