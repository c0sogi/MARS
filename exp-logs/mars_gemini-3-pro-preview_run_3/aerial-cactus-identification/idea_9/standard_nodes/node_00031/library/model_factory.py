import torch
import torch.nn as nn
import torchvision.models as models
import timm
from library.config import Config


def get_model(model_name, pretrained=True):
    """
    Constructs the specified model architecture with 'Stem Surgery' adaptations
    for 32x32 input images.

    The stem surgery replaces the initial aggressive downsampling (stride 2 conv + maxpool)
    with a stride 1 convolution and no pooling. This preserves the feature map size
    (32x32) entering the first residual block, which is critical for low-res imagery.

    Args:
        model_name (str): Name of the model ('resnet34', 'densenet121', 'seresnext50_32x4d').
        pretrained (bool): Whether to load ImageNet pretrained weights for the backbone.

    Returns:
        torch.nn.Module: The adapted PyTorch model.
    """

    if model_name == "resnet34":
        # Load ResNet34 from torchvision
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)

        # --- Stem Surgery ---
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New: Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Remove MaxPool to prevent downsampling from 32x32 to 16x16 immediately
        model.maxpool = nn.Identity()

        # --- Head Adaptation ---
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    elif model_name == "densenet121":
        # Load DenseNet121 from torchvision
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)

        # --- Stem Surgery ---
        # DenseNet features are contained in the 'features' Sequential block
        # conv0 is the initial 7x7 stride 2 convolution
        model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(
            model.features.conv0.weight, mode="fan_out", nonlinearity="relu"
        )

        # pool0 is the initial 3x3 stride 2 maxpool
        model.features.pool0 = nn.Identity()

        # --- Head Adaptation ---
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, Config.NUM_CLASSES)

    elif model_name == "seresnext50_32x4d":
        # Load SE-ResNeXt50 from timm
        # timm handles the classification head replacement via num_classes
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
        )

        # --- Stem Surgery ---
        # timm ResNet-based models typically expose 'conv1' and 'maxpool'
        if hasattr(model, "conv1"):
            in_channels = model.conv1.in_channels
            out_channels = model.conv1.out_channels
            model.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            nn.init.kaiming_normal_(
                model.conv1.weight, mode="fan_out", nonlinearity="relu"
            )

        if hasattr(model, "maxpool"):
            model.maxpool = nn.Identity()

    else:
        raise ValueError(f"Model '{model_name}' is not supported by model_factory.")

    return model
