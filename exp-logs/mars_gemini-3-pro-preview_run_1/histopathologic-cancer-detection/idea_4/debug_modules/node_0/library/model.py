import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model():
    """
    Constructs a ConvNeXt-Tiny model with a modified stem to preserve spatial resolution
    for small input images (48x48).

    Returns:
        torch.nn.Module: The modified ConvNeXt model.
    """
    # Instantiate the model using timm
    # We load ImageNet weights for transfer learning
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=Config.DROP_PATH_RATE,
    )

    # -------------------------------------------------------------------------
    # Stem Modification
    # -------------------------------------------------------------------------
    # Standard ConvNeXt Stem: Conv2d(3, dim, kernel_size=4, stride=4)
    # This reduces 48x48 input to 12x12 immediately, losing texture detail.
    #
    # Modified Stem: Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
    # This preserves the 48x48 resolution entering the first stage.
    # -------------------------------------------------------------------------

    # In timm, model.stem is usually a nn.Sequential(Conv2d, LayerNorm)
    if hasattr(model, "stem") and len(model.stem) > 0:
        old_conv = model.stem[0]

        if isinstance(old_conv, nn.Conv2d):
            # Create a new convolution layer
            # in_channels=3 (RGB), out_channels=model_dim
            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=(old_conv.bias is not None),
            )

            # Initialize the new layer's weights
            # Since this layer changes the input physics, we init from scratch
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="linear"
            )
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)

            # Replace the original downsampling conv with the new resolution-preserving conv
            model.stem[0] = new_conv

    return model
