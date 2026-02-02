import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

# ------------------------------------------------------------------------------
# 1. Custom Wide SE-ResNet
# ------------------------------------------------------------------------------


class SEBlock(nn.Module):
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


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.se(out)
        out += self.shortcut(x)
        return out


class WideSEResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, dropout_rate=0.3, num_classes=1):
        super(WideSEResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._wide_layer(
            WideBasicBlock, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            WideBasicBlock, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            WideBasicBlock, nStages[3], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ------------------------------------------------------------------------------
# 2. Custom DenseNet-BC (CIFAR Scale)
# ------------------------------------------------------------------------------


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNetBC(nn.Module):
    def __init__(self, depth=40, growth_rate=12, reduction=0.5, num_classes=1):
        super(DenseNetBC, self).__init__()
        n_blocks = (depth - 4) // 6

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(
            Bottleneck, num_planes, n_blocks, growth_rate
        )
        num_planes += n_blocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(
            Bottleneck, num_planes, n_blocks, growth_rate
        )
        num_planes += n_blocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(
            Bottleneck, num_planes, n_blocks, growth_rate
        )
        num_planes += n_blocks * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layers(self, block, in_planes, n_block, growth_rate):
        layers = []
        for i in range(n_block):
            layers.append(block(in_planes, growth_rate))
            in_planes += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ------------------------------------------------------------------------------
# 3. Modified EfficientNet-B0
# ------------------------------------------------------------------------------


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ModifiedEfficientNet, self).__init__()
        # Load standard EfficientNet-B0
        # We load with num_classes=0 to get the feature extractor, but timm handles
        # classifier replacement nicely if we just specify num_classes.
        # However, to modify the stem cleanly, we load the full model.
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
        )

        # Modify the stem (first convolution)
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Modified: Stride=1 to preserve 32x32 resolution
        original_stem = self.model.conv_stem
        self.model.conv_stem = nn.Conv2d(
            in_channels=original_stem.in_channels,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=1,  # Change stride to 1
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
        )

        # Initialize the new stem
        nn.init.kaiming_normal_(
            self.model.conv_stem.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.model.conv_stem.bias is not None:
            nn.init.constant_(self.model.conv_stem.bias, 0)

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------------


def get_model(model_name, num_classes=1, pretrained=False):
    """
    Factory function to instantiate models based on the configuration name.

    Args:
        model_name (str): The name of the model architecture.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights (where applicable).
                           Note: ModifiedEfficientNet will always re-init the stem.

    Returns:
        nn.Module: The requested model instance.
    """
    if model_name == "custom_wide_se_resnet":
        # WideResNet-16-4 with SE
        return WideSEResNet(
            depth=16, widen_factor=4, dropout_rate=0.3, num_classes=num_classes
        )

    elif model_name == "custom_densenet_bc":
        # DenseNet-BC-40-12
        return DenseNetBC(
            depth=40, growth_rate=12, reduction=0.5, num_classes=num_classes
        )

    elif model_name == "modified_efficientnet_b0":
        # EfficientNet-B0 with stride-1 stem
        return ModifiedEfficientNet(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
