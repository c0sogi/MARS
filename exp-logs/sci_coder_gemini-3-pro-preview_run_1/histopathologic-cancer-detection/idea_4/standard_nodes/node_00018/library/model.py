import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model():
    """
    Constructs a model (DenseNet121) with a modified stem to preserve spatial resolution
    for small input images (48x48).
    Cite solution_lesson_node_00016: Prefer DenseNet for small, texture-rich inputs.
    Cite solution_lesson_node_00008: Modify stem to prevent early downsampling.

    Returns:
        torch.nn.Module: The modified model.
    """
    # Instantiate the model using timm
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=Config.DROP_PATH_RATE,
    )

    # -------------------------------------------------------------------------
    # Stem Modification
    # -------------------------------------------------------------------------

    # Handle DenseNet121
    if "densenet" in Config.MODEL_NAME:
        # DenseNet stem: 7x7 stride 2 conv -> 3x3 stride 2 maxpool
        # We want: 3x3 stride 1 conv -> Identity pool

        # Access features container
        if hasattr(model, "features"):
            # 1. Modify Convolution (conv0)
            if hasattr(model.features, "conv0"):
                old_conv = model.features.conv0
                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(old_conv.bias is not None),
                )
                # Initialize
                nn.init.kaiming_normal_(
                    new_conv.weight, mode="fan_out", nonlinearity="relu"
                )
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)

                model.features.conv0 = new_conv

            # 2. Remove Pooling (pool0)
            if hasattr(model.features, "pool0"):
                model.features.pool0 = nn.Identity()

    # Handle ConvNeXt (Fallback if config changes back)
    elif hasattr(model, "stem") and len(model.stem) > 0:
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

    return model
