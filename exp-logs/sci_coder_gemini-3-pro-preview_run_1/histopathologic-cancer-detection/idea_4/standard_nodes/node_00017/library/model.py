import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model():
    """
    Constructs a model (DenseNet or ConvNeXt) with a modified stem to preserve spatial resolution
    for small input images (48x48). Cite solution_lesson_node_00003, solution_lesson_node_00008.

    Returns:
        torch.nn.Module: The modified model.
    """
    # Prepare kwargs
    model_kwargs = {
        "pretrained": Config.PRETRAINED,
        "num_classes": Config.NUM_CLASSES,
    }

    # Only pass drop_path_rate if the model supports it (e.g., ConvNeXt)
    # DenseNet does not support drop_path_rate in standard timm implementation
    if "convnext" in Config.MODEL_NAME:
        model_kwargs["drop_path_rate"] = Config.DROP_PATH_RATE

    # Instantiate the model using timm
    model = timm.create_model(Config.MODEL_NAME, **model_kwargs)

    # -------------------------------------------------------------------------
    # Stem Modification
    # -------------------------------------------------------------------------
    # Standard backbones (ResNet, DenseNet, ConvNeXt) typically downsample
    # aggressively in the first layer (stride 2 or 4). For 48x48 inputs, this
    # destroys information. We replace the first conv with stride 1 and remove
    # initial pooling.
    # -------------------------------------------------------------------------

    # Case 1: ConvNeXt (stem is usually a Sequential(Conv2d, LayerNorm))
    if hasattr(model, "stem") and len(model.stem) > 0:
        old_conv = model.stem[0]
        if isinstance(old_conv, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=(old_conv.bias is not None),
            )
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="linear"
            )
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
            model.stem[0] = new_conv

    # Case 2: DenseNet (features is a Sequential)
    # Standard DenseNet:
    #   features[0] = Conv2d(7x7, stride=2)
    #   features[3] = MaxPool2d(3x3, stride=2)
    elif hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        # 1. Modify First Convolution
        first_layer = model.features[0]
        if isinstance(first_layer, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=first_layer.in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=(first_layer.bias is not None),
            )
            # DenseNet uses ReLU, so init for ReLU
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="relu"
            )
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)
            model.features[0] = new_conv

        # 2. Remove Max Pooling (usually at index 3 for DenseNet121)
        # We replace it with Identity to keep the Sequential structure valid
        if len(model.features) > 3 and isinstance(model.features[3], nn.MaxPool2d):
            model.features[3] = nn.Identity()

    return model
