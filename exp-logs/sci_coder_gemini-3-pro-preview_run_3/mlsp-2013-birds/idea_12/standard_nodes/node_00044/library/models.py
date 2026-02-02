import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name: str,
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = Config.PRETRAINED,
):
    """
    Factory function to instantiate CNN backbones for bird species classification.

    This function leverages the `timm` library to create models with ImageNet pretrained weights.
    It automatically handles the modification of the classification head to match the target
    number of classes (19).

    Strategy Compliance:
    - Supports 'resnet18', 'efficientnet_b0', 'densenet121'.
    - Enforces 3-channel input (Config.INPUT_CHANNELS) to preserve the integrity of
      pretrained weights in the first convolutional layer, matching the data replication strategy.

    Args:
        model_name (str): The name of the architecture (e.g., 'resnet18').
        num_classes (int): The number of output logits. Defaults to 19.
        pretrained (bool): Whether to load ImageNet weights. Defaults to True.

    Returns:
        nn.Module: The instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not supported by timm or instantiation fails.
    """

    # Although the system is designed for specific models, we allow any valid timm name
    # for flexibility during potential future experiments.
    # The Config.MODELS list acts as the approved registry.

    try:
        # Instantiate model using timm
        # in_chans=3 is explicitly passed to ensure the model expects the pseudo-RGB
        # spectrograms created in the Dataset class.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=Config.INPUT_CHANNELS,
        )

    except Exception as e:
        raise ValueError(f"Failed to create model '{model_name}'. Error: {e}")

    return model
