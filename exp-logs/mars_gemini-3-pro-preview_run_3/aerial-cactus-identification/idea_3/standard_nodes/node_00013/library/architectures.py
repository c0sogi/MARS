import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# SHARED COMPONENTS
# =============================================================================


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =============================================================================
# WIDE SE-RESNET
# =============================================================================


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_rate=0.0, se_reduction=16):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = drop_rate
        self.equalInOut = in_planes == planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            or None
        )

        self.se = SEBlock(planes, reduction=se_reduction)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)

        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        out = self.se(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideSEResNet(nn.Module):
    def __init__(
        self,
        depth,
        widen_factor,
        drop_rate=0.0,
        num_classes=1,
        input_channels=3,
        se_reduction=16,
        stem_type="cifar",
    ):
        super(WideSEResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "Depth must be 6n + 4"
        n = (depth - 4) // 6

        # Stem adaptation
        if stem_type == "cifar":
            # Preserves 32x32 resolution
            self.conv1 = nn.Conv2d(
                input_channels,
                nChannels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        else:
            # Standard ImageNet stem (downsamples significantly)
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    nChannels[0],
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(nChannels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.block1 = self._make_layer(
            n,
            nChannels[0],
            nChannels[1],
            stride=1,
            drop_rate=drop_rate,
            se_reduction=se_reduction,
        )
        self.block2 = self._make_layer(
            n,
            nChannels[1],
            nChannels[2],
            stride=2,
            drop_rate=drop_rate,
            se_reduction=se_reduction,
        )
        self.block3 = self._make_layer(
            n,
            nChannels[2],
            nChannels[3],
            stride=2,
            drop_rate=drop_rate,
            se_reduction=se_reduction,
        )

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, n, in_planes, out_planes, stride, drop_rate, se_reduction):
        layers = []
        for i in range(n):
            layers.append(
                WideBasicBlock(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                    se_reduction,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


# =============================================================================
# DENSENET-BC
# =============================================================================


class DenseLayer(nn.Module):
    def __init__(self, in_planes, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseTransition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DenseTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNetBC(nn.Module):
    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        compression=0.5,
        num_init_features=24,
        bn_size=4,
        drop_rate=0.0,
        num_classes=1,
        input_channels=3,
        stem_type="cifar",
    ):
        super(DenseNetBC, self).__init__()

        # Stem adaptation
        if stem_type == "cifar":
            self.features = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    num_init_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(
                num_layers, num_features, growth_rate, bn_size, drop_rate
            )
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = DenseTransition(num_features, out_features)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        self.classifier = nn.Linear(num_features, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, num_layers, in_planes, growth_rate, bn_size, drop_rate):
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(in_planes + i * growth_rate, growth_rate, bn_size, drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return out
