import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import INPUT_SHAPE, CHANNEL_CONFIG, NUM_CLASSES


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block.
    """

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
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiScaleResNet(nn.Module):
    """
    Multi-Scale ResNet with GAP Aggregation (Cite solution_lesson_node_00016).
    """

    def __init__(self):
        super(MultiScaleResNet, self).__init__()

        # Channel Configuration: [16, 32, 64]
        c1, c2, c3 = CHANNEL_CONFIG

        # Stem: 32x32 input -> 32x32 feature map
        self.conv1 = nn.Conv2d(
            INPUT_SHAPE[0], c1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(c1)

        # Stage 1: 16 channels, 32x32 resolution
        self.layer1 = self._make_layer(c1, c1, stride=1)

        # Stage 2: 32 channels, 16x16 resolution
        self.layer2 = self._make_layer(c1, c2, stride=2)

        # Stage 3: 64 channels, 8x8 resolution
        self.layer3 = self._make_layer(c2, c3, stride=2)

        # --- Multi-Scale Aggregation ---
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Concat GAP(Stage 2) + GAP(Stage 3) -> 32 + 64 = 96
        self.fc = nn.Linear(c2 + c3, NUM_CLASSES)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))

        # Stage 1
        out = self.layer1(out)

        # Stage 2
        stage2_out = self.layer2(out)  # (B, 32, 16, 16)

        # Stage 3
        stage3_out = self.layer3(stage2_out)  # (B, 64, 8, 8)

        # --- Aggregation ---
        # GAP on Stage 2
        feat2 = self.gap(stage2_out).flatten(1)  # (B, 32)

        # GAP on Stage 3
        feat3 = self.gap(stage3_out).flatten(1)  # (B, 64)

        # Fusion
        fused_feat = torch.cat([feat2, feat3], dim=1)  # (B, 96)

        # Classification
        logits = self.fc(fused_feat)

        return logits
