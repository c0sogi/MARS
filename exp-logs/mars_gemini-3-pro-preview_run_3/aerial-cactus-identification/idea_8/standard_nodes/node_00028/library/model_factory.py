import torch
import torch.nn as nn
import timm


def modify_stem(model, model_name):
    """
    Modifies the stem of the model to handle 32x32 inputs without aggressive downsampling.
    Replaces 7x7 stride-2 conv with 3x3 stride-1 conv.
    Removes the initial maxpooling layer.

    Args:
        model (nn.Module): The model to modify.
        model_name (str): The name of the architecture.

    Returns:
        nn.Module: The modified model.
    """
    # Logic for ResNet / SE-ResNeXt architectures
    # These typically have self.conv1 (7x7 s2) and self.maxpool (3x3 s2)
    if any(x in model_name for x in ["resnet", "resnext", "seresnext"]):
        # 1. Replace Conv1 (7x7 s2 -> 3x3 s1)
        if hasattr(model, "conv1"):
            in_channels = model.conv1.in_channels
            out_channels = model.conv1.out_channels
            # Create new layer (randomly initialized by default)
            new_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.conv1 = new_conv

        # 2. Remove MaxPool (3x3 s2 -> Identity)
        if hasattr(model, "maxpool"):
            model.maxpool = nn.Identity()

    # Logic for DenseNet architectures
    # These typically have self.features.conv0 and self.features.pool0
    elif "densenet" in model_name:
        if hasattr(model, "features"):
            features = model.features
            # 1. Replace Conv0
            if hasattr(features, "conv0"):
                in_channels = features.conv0.in_channels
                out_channels = features.conv0.out_channels
                new_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                features.conv0 = new_conv

            # 2. Remove Pool0
            if hasattr(features, "pool0"):
                features.pool0 = nn.Identity()

    return model


def get_model(model_name, num_classes=1, pretrained=True, stem_surgery=True):
    """
    Creates a model using timm, optionally applying stem surgery.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet34').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load ImageNet pretrained weights.
        stem_surgery (bool): Whether to modify the stem for 32x32 inputs.

    Returns:
        nn.Module: The constructed model.
    """
    # Create model using timm
    # We load pretrained weights here. If stem_surgery is True,
    # the stem layers will be replaced with fresh random weights immediately after.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    if stem_surgery:
        model = modify_stem(model, model_name)

    return model
