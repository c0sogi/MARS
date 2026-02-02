import torch
import torch.nn as nn
import math
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adaptively recalibrates channel-wise feature responses.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure reduction doesn't make channels < 1
        reduced_channel = max(1, channel // reduction)
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


class Res2NeXtBottleneck(nn.Module):
    """
    Custom Bottleneck integrating Res2Net hierarchical connections and ResNeXt grouped convolutions.
    """

    expansion = 1  # Wide configuration: Internal width matches output width

    def __init__(
        self, inplanes, planes, stride=1, scales=4, groups=32, se_reduction=16
    ):
        super(Res2NeXtBottleneck, self).__init__()

        self.scales = scales
        self.stride = stride
        self.planes = planes

        # Calculate width per scale
        # We ensure the total width is divisible by scales
        width = planes
        if width % scales != 0:
            # In case of mismatch, we could raise error or adjust.
            # With config [64, 128, 256] and scale 4, it is valid.
            pass

        self.width_per_scale = width // scales

        # 1x1 Expansion/Projection
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 Hierarchical Grouped Convolutions
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Determine valid group count for the 3x3 convs
        # Must divide width_per_scale and be <= width_per_scale
        # Also try to match requested cardinality
        effective_groups = min(groups, self.width_per_scale)
        while self.width_per_scale % effective_groups != 0:
            effective_groups -= 1

        # We need (scales - 1) convs
        self.nums = scales - 1
        for i in range(self.nums):
            self.convs.append(
                nn.Conv2d(
                    self.width_per_scale,
                    self.width_per_scale,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=effective_groups,
                    bias=False,
                )
            )
            self.bns.append(nn.BatchNorm2d(self.width_per_scale))

        # Handling the first split (identity or pool) for stride > 1
        self.pool = None
        if stride > 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        # 1x1 Projection to Output
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction=se_reduction)

        # Shortcut connection
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            # Strictly use 1x1 conv for shortcut
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Split features into 'scales' chunks
        xs = torch.chunk(out, self.scales, 1)
        ys = []

        # Process first chunk
        # If stride > 1, we must downsample x1 to match spatial dims of other branches
        if self.stride > 1:
            y1 = self.pool(xs[0])
        else:
            y1 = xs[0]
        ys.append(y1)

        # Process subsequent chunks
        prev = y1
        for i in range(self.nums):
            # If stride is 1, we add the previous output (hierarchical connection)
            # If stride > 1, we break the connection to avoid size mismatch (xs is large, prev is small)
            if self.stride == 1:
                x_in = xs[i + 1] + prev
            else:
                x_in = xs[i + 1]

            z = self.convs[i](x_in)
            z = self.bns[i](z)
            z = self.relu(z)
            ys.append(z)
            prev = z

        # Concatenate
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WideSERes2NeXt(nn.Module):
    """
    Wide SE-Res2NeXt with Multi-Scale Aggregation.
    """

    def __init__(self):
        super(WideSERes2NeXt, self).__init__()

        # Load config
        conf = Config.MODEL_CONFIG
        self.stage_channels = conf.get("stage_channels", [64, 128, 256])
        self.cardinality = conf.get("cardinality", 32)
        self.scale = conf.get("res2net_scale", 4)
        self.se_reduction = conf.get("se_reduction", 16)
        self.num_classes = conf.get("num_classes", 1)

        # Initial Stem
        # Input 32x32 -> Conv 3x3 -> 32x32
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: 32x32 output
        self.layer1 = self._make_layer(self.stage_channels[0], blocks=3, stride=1)

        # Stage 2: 16x16 output
        self.layer2 = self._make_layer(self.stage_channels[1], blocks=3, stride=2)

        # Stage 3: 8x8 output
        self.layer3 = self._make_layer(self.stage_channels[2], blocks=3, stride=2)

        # Multi-Scale Aggregation Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate final dimension
        # We concatenate features from Stage 2 and Stage 3
        dim2 = self.stage_channels[1] * Res2NeXtBottleneck.expansion
        dim3 = self.stage_channels[2] * Res2NeXtBottleneck.expansion

        self.fc = nn.Linear(dim2 + dim3, self.num_classes)

        # Initialization
        self._init_weights()

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        # First block
        layers.append(
            Res2NeXtBottleneck(
                self.inplanes,
                planes,
                stride,
                scales=self.scale,
                groups=self.cardinality,
                se_reduction=self.se_reduction,
            )
        )
        self.inplanes = planes * Res2NeXtBottleneck.expansion

        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(
                Res2NeXtBottleneck(
                    self.inplanes,
                    planes,
                    1,
                    scales=self.scale,
                    groups=self.cardinality,
                    se_reduction=self.se_reduction,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Stages
        x1 = self.layer1(x)  # Stage 1 (32x32)
        x2 = self.layer2(x1)  # Stage 2 (16x16)
        x3 = self.layer3(x2)  # Stage 3 (8x8)

        # Multi-Scale Aggregation
        # GAP on Stage 2
        pool2 = self.avgpool(x2).flatten(1)
        # GAP on Stage 3
        pool3 = self.avgpool(x3).flatten(1)

        # Concatenate
        combined = torch.cat([pool2, pool3], dim=1)

        # Classifier
        out = self.fc(combined)

        return out
