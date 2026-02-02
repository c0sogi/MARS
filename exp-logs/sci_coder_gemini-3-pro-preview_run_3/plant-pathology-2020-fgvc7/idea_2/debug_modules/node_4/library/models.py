import timm
import torch.nn as nn
from library.config import NUM_CLASSES


def get_model(model_name, pretrained=True, num_classes=NUM_CLASSES):
    """
    Instantiates a model architecture using timm and configures the final classification head.

    This function handles the creation of both EfficientNet and ConvNeXt models as specified
    in the configuration. It automatically replaces the original classification head with
    one matching the target number of classes.

    Args:
        model_name (str): The name of the model architecture to load
                          (e.g., 'tf_efficientnet_b0_ns', 'convnext_tiny').
        pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.
        num_classes (int): The number of output classes for the classification head.
                           Defaults to NUM_CLASSES from config.

    Returns:
        nn.Module: The PyTorch model with the modified head ready for training or inference.
    """
    # Create the model using timm.
    # Specifying num_classes triggers timm to replace the pre-trained head
    # with a new randomly initialized head of the correct size.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
