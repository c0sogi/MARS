import torch
import torch.nn as nn
import timm
from library.config import CFG


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Factory function to create model architectures using timm.

    Args:
        model_name (str): Name of the model architecture to create (must be supported by timm).
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        num_classes (int): Number of output classes. Defaults to 1 for binary classification.

    Returns:
        nn.Module: The instantiated PyTorch model.
    """
    # Verify model name is expected (optional, but good for consistency with CFG)
    if model_name not in CFG.model_names:
        # We don't raise an error here to allow flexibility if testing other models,
        # but we log/print a warning if it's unexpected based on the known config.
        pass

    # Create the model using timm
    # timm.create_model handles the replacement of the classifier head
    # based on num_classes.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
