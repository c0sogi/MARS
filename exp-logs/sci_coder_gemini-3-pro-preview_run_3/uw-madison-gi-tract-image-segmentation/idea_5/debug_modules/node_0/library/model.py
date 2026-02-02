import torch
import segmentation_models_pytorch as smp
from library.config import CFG


def build_model():
    """
    Builds the U-Net++ (Nested U-Net) model with an EfficientNet-B4 backbone.

    The model is configured based on the parameters in library.config.CFG:
    - Architecture: U-Net++
    - Encoder: timm-efficientnet-b4
    - Pretrained Weights: noisy-student
    - Input Channels: 3 (2.5D context: t-1, t, t+1)
    - Output Classes: 3 (Large Bowel, Small Bowel, Stomach)
    - Activation: None (returns logits for BCEWithLogitsLoss)

    Returns:
        torch.nn.Module: The instantiated U-Net++ model.
    """
    model = smp.UnetPlusPlus(
        encoder_name=CFG.backbone,  # e.g., "timm-efficientnet-b4"
        encoder_weights=CFG.encoder_weights,  # e.g., "noisy-student"
        in_channels=CFG.in_chans,  # 3
        classes=CFG.n_classes,  # 3
        activation=None,  # Return logits
    )

    return model
