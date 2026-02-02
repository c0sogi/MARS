import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from library.config import Config

# -----------------------------------------------------------------------------
# 1. Modified Wide SE-ResNet
# -----------------------------------------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
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
        return x * y


class BasicBlock(nn.Module):
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
        self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModifiedWideSEResNet(nn.Module):
    def __init__(self, num_classes=1, widen_factor=4):
        super(ModifiedWideSEResNet, self).__init__()
        self.in_planes = 16

        # Modified Stem: 3x3 Conv, stride 1 to preserve 32x32 resolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Layers with increased width
        self.layer1 = self._make_layer(16 * widen_factor, 2, stride=1)
        self.layer2 = self._make_layer(32 * widen_factor, 2, stride=2)
        self.layer3 = self._make_layer(64 * widen_factor, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# -----------------------------------------------------------------------------
# 2. Modified DenseNet-BC
# -----------------------------------------------------------------------------


class ModifiedDenseNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedDenseNet, self).__init__()
        # Load standard DenseNet121
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Modify Stem for 32x32 input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove MaxPool0 to preserve spatial dimensions early on
        self.model.features.pool0 = nn.Identity()

        # Modify Classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# 3. Modified EfficientNet
# -----------------------------------------------------------------------------


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedEfficientNet, self).__init__()
        # Use timm to create EfficientNet B0
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

        # Modify Stem for 32x32 input
        # Original stem usually has stride 2. We want stride 1.
        original_stem = self.model.conv_stem
        out_channels = original_stem.out_channels

        # Create new stem with stride 1
        self.model.conv_stem = nn.Conv2d(
            3, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Do not copy weights when changing stride from 2 to 1 for low-res inputs.
        # Random initialization allows the stem to learn appropriate low-level filters
        # for the new resolution (Cite solution_lesson_node_00023).

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_model(model_name, num_classes=1):
    """
    Factory function to instantiate models by name.

    Args:
        model_name (str): Name of the model architecture.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The requested model.
    """
    if model_name == "wide_se_resnet":
        return ModifiedWideSEResNet(num_classes=num_classes)
    elif model_name == "densenet_bc":
        return ModifiedDenseNet(num_classes=num_classes)
    elif model_name == "efficientnet":
        return ModifiedEfficientNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
