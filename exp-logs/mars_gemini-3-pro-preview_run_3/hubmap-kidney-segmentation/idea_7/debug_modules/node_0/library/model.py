import torch
import segmentation_models_pytorch as smp
from library.config import Config


def build_model():
    """
    Constructs the U-Net++ model with a ConvNeXt-Tiny backbone.

    Uses parameters from the Config class:
    - Architecture: U-Net++
    - Encoder: tu-convnext_tiny (via timm)
    - Pretrained Weights: ImageNet
    - Deep Supervision: Enabled (returns list of outputs from decoder levels)
    - Decoder Channels: Custom list [256, 128, 64, 32, 16]

    Returns:
        torch.nn.Module: The instantiated PyTorch model.
    """

    # Instantiate the U-Net++ model
    # Note: When deep_supervision=True, the model forward pass returns a list of tensors
    # corresponding to the outputs of the decoder stages.
    model = smp.UnetPlusPlus(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
        activation=Config.ACTIVATION,  # None, using BCEWithLogitsLoss
        decoder_channels=Config.DECODER_CHANNELS,
        deep_supervision=Config.DEEP_SUPERVISION,
    )

    return model
