import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block with 3x3 convolutions.
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LightweightPyramidNet(nn.Module):
    """
    Custom Lightweight Feature Pyramid Classifier.

    Architecture:
    - Backbone: Narrow ResNet (16, 32, 64 channels) with 3 stages.
    - Neck: Spatial Feature Fusion at 16x16 resolution.
    - Head: Global Average Pooling + Linear Classifier.
    """

    def __init__(self):
        super(LightweightPyramidNet, self).__init__()
        self.in_planes = 16

        # --- Backbone ---
        # Initial convolution: 32x32 input -> 32x32 feature map
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1: 16 channels, 32x32 output
        self.layer1 = self._make_layer(16, 2, stride=1)

        # Stage 2: 32 channels, 16x16 output (downsample)
        self.layer2 = self._make_layer(32, 2, stride=2)

        # Stage 3: 64 channels, 8x8 output (downsample)
        self.layer3 = self._make_layer(64, 2, stride=2)

        # --- Neck ---
        # Project Stage 3 (64ch) -> 32ch
        self.lat_s3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)

        # Project Stage 2 (32ch) -> 32ch
        self.lat_s2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)

        # --- Head ---
        # Classifier: 32 input features -> 1 output logit
        self.fc = nn.Linear(32, 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- Backbone Forward ---
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32

        c1 = self.layer1(out)  # Stage 1: 32x32, 16ch
        c2 = self.layer2(c1)  # Stage 2: 16x16, 32ch
        c3 = self.layer3(c2)  # Stage 3: 8x8, 64ch

        # --- Neck Forward ---
        # Process Stage 3: Project and Upsample
        p3 = self.lat_s3(c3)  # 8x8, 32ch
        p3_up = F.interpolate(p3, scale_factor=2, mode="nearest")  # 16x16, 32ch

        # Process Stage 2: Project
        p2 = self.lat_s2(c2)  # 16x16, 32ch

        # Fusion: Element-wise addition
        fused = p2 + p3_up  # 16x16, 32ch

        # --- Head Forward ---
        out = F.adaptive_avg_pool2d(fused, (1, 1))  # Global Average Pooling
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)  # Linear Classifier

        return out
