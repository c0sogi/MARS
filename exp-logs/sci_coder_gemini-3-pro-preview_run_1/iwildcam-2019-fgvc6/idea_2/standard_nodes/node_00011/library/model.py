import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, NUM_CLASSES, PRETRAINED


def get_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """
    Initializes and returns a model with a custom classification head.

    This function uses the `timm` library to load an architecture.
    If `pretrained` is True, it loads weights pre-trained on ImageNet-1k.
    The final classification layer is automatically replaced by `timm` to match
    the specified `num_classes`.

    Args:
        model_name (str): The name of the architecture to use (e.g., 'convnext_tiny').
                          Defaults to MODEL_NAME from config.
        num_classes (int): The number of target classes. Defaults to NUM_CLASSES from config.
        pretrained (bool): Whether to load pretrained weights. Defaults to PRETRAINED from config.

    Returns:
        torch.nn.Module: The PyTorch model ready for training.
    """
    try:
        # Create the model using timm.
        # When num_classes is specified and different from the pretrained model's default (usually 1000),
        # timm automatically replaces the classifier head (fc layer) with a new one
        # initialized with random weights and the correct output dimension.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        return model

    except Exception as e:
        print(f"Error initializing model '{model_name}': {e}")
        raise e
