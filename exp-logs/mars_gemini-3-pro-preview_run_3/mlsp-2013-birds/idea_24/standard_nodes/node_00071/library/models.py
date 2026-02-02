import torch
import torch.nn as nn
import timm
from library.utils import seed_everything


def get_bird_model(model_name, num_classes=19, pretrained=True):
    """
    Factory function to initialize a bird species classification model.

    This function leverages the `timm` library to create models with pre-trained
    ImageNet weights. It automatically replaces the classification head with a
    linear layer outputting `num_classes` logits. It strictly enforces a
    3-channel input configuration to match the pseudo-RGB spectrograms.

    Supported Architectures (passed as model_name):
    - 'resnet18'
    - 'efficientnet_b0'
    - 'densenet121'

    Args:
        model_name (str): The name of the backbone architecture.
        num_classes (int): The number of output classes (default: 19).
        pretrained (bool): Whether to load pre-trained ImageNet weights (default: True).

    Returns:
        torch.nn.Module: The configured PyTorch model ready for training/inference.
    """
    # Ensure deterministic initialization for the new head
    seed_everything(42)

    try:
        # Create the model using timm
        # - pretrained=True: Loads weights from ImageNet
        # - num_classes=19: Replaces the head with a Linear(in_features, 19)
        # - in_chans=3: Ensures the input layer accepts 3-channel images
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )
    except Exception as e:
        raise ValueError(
            f"Failed to create model '{model_name}'. "
            f"Ensure specific model name is supported by timm. Error: {e}"
        )

    return model
