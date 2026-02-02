import timm
import torch.nn as nn
from library.config import Config


def get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Creates and returns a model architecture based on the provided name using the timm library.

    This function instantiates the backbone (e.g., ResNet101d, ConvNeXt-Base) and
    replaces the classification head to match the specific number of classes in the dataset.
    It adheres to the strategy of using Global Average Pooling followed by a Linear projection,
    which is the default behavior of timm's create_model when num_classes is specified.

    Args:
        model_name (str): Name of the model architecture (e.g., 'resnet101d', 'convnext_base').
        num_classes (int): Number of output classes (attributes). Defaults to Config.NUM_CLASSES.
        pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.

    Returns:
        nn.Module: The instantiated PyTorch model.
    """
    try:
        # Instantiate the model using timm
        # num_classes argument automatically resets the classifier head to the correct size
        # and initializes it randomly.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}' using timm: {e}")
