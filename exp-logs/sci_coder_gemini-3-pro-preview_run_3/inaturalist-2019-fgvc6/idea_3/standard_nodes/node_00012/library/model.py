import torch
import timm
from library.config import Config


def get_model(
    num_classes=Config.NUM_CLASSES, pretrained=True, model_name=Config.MODEL_NAME
):
    """
    Initializes the ConvNeXt-Base model using timm.

    Args:
        num_classes (int): Number of output classes (default: 1010).
        pretrained (bool): Whether to load pretrained ImageNet-21k weights.
        model_name (str): Name of the model architecture in timm.

    Returns:
        model (torch.nn.Module): The instantiated model on the configured device.
    """
    # Create the model using timm
    # passing num_classes tells timm to replace the head with a new one
    # initialized for the specific number of classes.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    # Move the model to the appropriate device (GPU or CPU)
    model = model.to(Config.DEVICE)

    return model
