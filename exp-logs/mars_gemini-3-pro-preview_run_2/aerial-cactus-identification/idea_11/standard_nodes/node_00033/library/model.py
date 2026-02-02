import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adaptively recalibrates channel-wise feature responses.
    """

    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        # Ensure the bottleneck has at least 1 channel
        reduced_channel = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channel, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block with optional SE Module.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE -> Add -> ReLU
    """

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=True):
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

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes * self.expansion)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DualStreamHead(nn.Module):
    """
    Dual-Stream Multi-Scale Aggregation Head.
    Extracts features from Stage 2 and Stage 3, applies both GAP and GMP,
    concatenates them, and classifies.
    """

    def __init__(self, s2_channels, s3_channels, num_classes):
        super(DualStreamHead, self).__init__()
        # Input dimension = (Stage2_GAP + Stage2_GMP + Stage3_GAP + Stage3_GMP)
        self.input_dim = (s2_channels * 2) + (s3_channels * 2)
        self.fc = nn.Linear(self.input_dim, num_classes)

    def forward(self, s2_feat, s3_feat):
        # Stage 2 Aggregation
        s2_gap = F.adaptive_avg_pool2d(s2_feat, 1).flatten(1)
        s2_gmp = F.adaptive_max_pool2d(s2_feat, 1).flatten(1)

        # Stage 3 Aggregation
        s3_gap = F.adaptive_avg_pool2d(s3_feat, 1).flatten(1)
        s3_gmp = F.adaptive_max_pool2d(s3_feat, 1).flatten(1)

        # Concatenate all features
        combined = torch.cat([s2_gap, s2_gmp, s3_gap, s3_gmp], dim=1)

        return self.fc(combined)


class NarrowSEResNet(nn.Module):
    """
    Custom Narrow SE-ResNet with Dual-Stream Multi-Scale Aggregation.
    Backbone: 3 Stages with [16, 32, 64] channels.
    """

    def __init__(self):
        super(NarrowSEResNet, self).__init__()
        self.in_planes = Config.BACKBONE_CHANNELS[0]

        # Initial Stem
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Backbone Stages
        # Stage 1: 32x32 resolution
        self.layer1 = self._make_layer(
            BasicBlock, Config.BACKBONE_CHANNELS[0], num_blocks=2, stride=1
        )
        # Stage 2: 16x16 resolution
        self.layer2 = self._make_layer(
            BasicBlock, Config.BACKBONE_CHANNELS[1], num_blocks=2, stride=2
        )
        # Stage 3: 8x8 resolution
        self.layer3 = self._make_layer(
            BasicBlock, Config.BACKBONE_CHANNELS[2], num_blocks=2, stride=2
        )

        # Classification Head
        self.use_dual_stream = Config.USE_DUAL_STREAM_HEAD

        if self.use_dual_stream:
            self.head = DualStreamHead(
                s2_channels=Config.BACKBONE_CHANNELS[1],
                s3_channels=Config.BACKBONE_CHANNELS[2],
                num_classes=Config.NUM_CLASSES,
            )
        else:
            # Fallback to standard GAP head
            self.head = nn.Linear(Config.BACKBONE_CHANNELS[2], Config.NUM_CLASSES)

        # Weight Initialization
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, use_se=Config.USE_SE_BLOCKS)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        out = F.relu(self.bn1(self.conv1(x)))

        # Stage 1
        out = self.layer1(out)

        # Stage 2
        s2_feat = self.layer2(out)

        # Stage 3
        s3_feat = self.layer3(s2_feat)

        if self.use_dual_stream:
            logits = self.head(s2_feat, s3_feat)
        else:
            out = F.adaptive_avg_pool2d(s3_feat, 1).flatten(1)
            logits = self.head(out)

        return logits
