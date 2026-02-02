import torch
import torch.nn as nn
import torchvision
import timm


class ModifiedDenseNet(nn.Module):
    """
    Modified DenseNet-121 architecture.
    Replaces the initial 7x7 stride-2 stem and max pooling with a 3x3 stride-1 convolution
    to preserve spatial resolution for 32x32 inputs.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Use "DEFAULT" weights for the best available pretrained weights
        weights = "DEFAULT" if pretrained else None
        self.model = torchvision.models.densenet121(weights=weights)

        # 1. Modify Stem Convolution
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 2. Remove Stem Pooling
        # Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
        # This prevents the immediate 4x reduction (2x from conv, 2x from pool)
        self.model.features.pool0 = nn.Identity()

        # 3. Modify Classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ModifiedEfficientNet(nn.Module):
    """
    Modified EfficientNet-B0 architecture (Inverted Residuals).
    Replaces the initial stride-2 stem with a stride-1 convolution.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = torchvision.models.efficientnet_b0(weights=weights)

        # 1. Modify Stem Convolution
        # The first layer in EfficientNet is within features[0] (Conv2dNormActivation)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        first_conv_block = self.model.features[0]
        original_conv = first_conv_block[0]

        # Replace with stride 1
        first_conv_block[0] = nn.Conv2d(
            3,
            original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 2. Modify Classifier
        # EfficientNet classifier is Sequential(Dropout, Linear)
        dropout_layer = self.model.classifier[0]
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            dropout_layer, nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class ModifiedSEResNet(nn.Module):
    """
    Modified Wide SE-ResNet architecture.
    Uses SE-ResNeXt50-32x4d (Wide + SE + ResNeXt) from timm.
    Replaces the initial stem to preserve resolution.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        # Load SE-ResNeXt50 (32x4d indicates wide cardinality)
        self.model = timm.create_model(
            "seresnext50_32x4d", pretrained=pretrained, num_classes=num_classes
        )

        # 1. Modify Stem Convolution
        # timm ResNet models typically use 'conv1' for the first layer
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 2. Remove Stem Pooling
        # Original: MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ModifiedMobileNet(nn.Module):
    """
    Modified MobileNetV3-Large architecture.
    Replaces the initial stride-2 stem with a stride-1 convolution.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.model = torchvision.models.mobilenet_v3_large(weights=weights)

        # 1. Modify Stem Convolution
        # The first layer is features[0][0] (Conv2d within Conv2dNormActivation)
        # Original: Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        first_conv_block = self.model.features[0]
        original_conv = first_conv_block[0]

        first_conv_block[0] = nn.Conv2d(
            3,
            original_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 2. Modify Classifier
        # Classifier is Sequential(Linear, Hardswish, Dropout, Linear)
        last_linear = self.model.classifier[-1]
        in_features = last_linear.in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(model_name, num_classes=1, pretrained=True):
    """
    Factory function to instantiate the requested model architecture.

    Args:
        model_name (str): 'densenet', 'efficientnet', or 'resnet'.
        num_classes (int): Number of output classes (default 1 for binary).
        pretrained (bool): Whether to load ImageNet pretrained weights.

    Returns:
        nn.Module: The initialized model.
    """
    if model_name == "densenet":
        return ModifiedDenseNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "efficientnet":
        return ModifiedEfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet":
        return ModifiedSEResNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "mobilenet":
        return ModifiedMobileNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
