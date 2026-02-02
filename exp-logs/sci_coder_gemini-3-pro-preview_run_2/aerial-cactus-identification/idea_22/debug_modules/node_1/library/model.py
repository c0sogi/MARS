import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library import config


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Res2NetBottleneck(nn.Module):
    """
    Res2Net Bottleneck Block with Squeeze-and-Excitation.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        scale=4,
        base_width=26,
        use_se=True,
    ):
        super(Res2NetBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64.0))) * scale
        self.scale = scale
        self.width = width
        self.stride = stride
        self.use_se = use_se

        # The number of channels for each split
        self.nums = width // scale

        # 1x1 conv to reduce dimensions
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 convs for the splits
        # We need scale-1 convs because the first split is identity (or pool)
        convs = []
        bns = []
        for i in range(self.scale - 1):
            convs.append(
                nn.Conv2d(
                    self.nums,
                    self.nums,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
            bns.append(nn.BatchNorm2d(self.nums))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # If stride > 1, the identity branch (split 0) needs pooling to match spatial dims
        self.pool = (
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
            if stride > 1
            else None
        )

        # 1x1 conv to expand
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.use_se:
            self.se = SELayer(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Split features into 'scale' subsets
        spx = torch.split(out, self.nums, 1)

        # Process the first split (identity or pool)
        sp = spx[0]
        if self.pool:
            sp = self.pool(sp)

        outs = [sp]

        # Process remaining splits hierarchically
        for i in range(self.scale - 1):
            if i == 0:
                sp = self.convs[i](spx[i + 1])
            else:
                sp = self.convs[i](spx[i + 1] + sp)
            sp = self.relu(self.bns[i](sp))
            outs.append(sp)

        # Concatenate
        out = torch.cat(outs, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideSERes2Net(nn.Module):
    """
    Custom Wide SE-Res2Net with Multi-Stage Feature Fusion.
    """

    def __init__(self):
        super(WideSERes2Net, self).__init__()

        # Configuration
        channels = config.CHANNELS  # [64, 128, 256]
        scale = config.RES2NET_SCALE
        base_width = config.RES2NET_BASE_WIDTH
        use_se = config.USE_SE

        self.inplanes = 64

        # Initial Conv: 32x32 -> 32x32 (Preserve resolution)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: planes=64, stride=1. Output: 32x32.
        self.layer1 = self._make_layer(
            Res2NetBottleneck,
            channels[0],
            blocks=3,
            stride=1,
            scale=scale,
            base_width=base_width,
            use_se=use_se,
        )

        # Stage 2: planes=128, stride=2. Output: 16x16.
        self.layer2 = self._make_layer(
            Res2NetBottleneck,
            channels[1],
            blocks=3,
            stride=2,
            scale=scale,
            base_width=base_width,
            use_se=use_se,
        )

        # Stage 3: planes=256, stride=2. Output: 8x8.
        self.layer3 = self._make_layer(
            Res2NetBottleneck,
            channels[2],
            blocks=3,
            stride=2,
            scale=scale,
            base_width=base_width,
            use_se=use_se,
        )

        # Head: Multi-Stage Feature Fusion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate fusion dimension
        # Stage 2 output: planes=128, expansion=4 -> 512 channels
        # Stage 3 output: planes=256, expansion=4 -> 1024 channels
        fusion_dim = (channels[1] * 4) + (channels[2] * 4)

        self.fc = nn.Linear(fusion_dim, 1)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, scale=4, base_width=26, use_se=True
    ):
        downsample = None
        # Downsample if stride != 1 or inplanes != planes * expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                scale=scale,
                base_width=base_width,
                use_se=use_se,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    scale=scale,
                    base_width=base_width,
                    use_se=use_se,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)  # 32x32

        # Stage 2
        feat2 = self.layer2(x)  # 16x16

        # Stage 3
        feat3 = self.layer3(feat2)  # 8x8

        # Fusion Head
        pool2 = self.avgpool(feat2).flatten(1)
        pool3 = self.avgpool(feat3).flatten(1)

        concat = torch.cat([pool2, pool3], dim=1)

        out = self.fc(concat)

        return out
