import torch
import timm
from library import config


def get_model(
    model_name=config.MODEL_NAME,
    num_classes=config.NUM_CLASSES,
    pretrained=config.PRETRAINED,
    device=None,
):
    """
    Instantiates a ConvNeXt model using the timm library, modifies the classifier head
    for the specific number of classes, and moves it to the specified device.

    Args:
        model_name (str): The name of the model architecture (e.g., 'convnext_small.fb_in1k').
        num_classes (int): The number of output classes for the new head.
        pretrained (bool): Whether to load pretrained ImageNet weights.
        device (str or torch.device, optional): The device to move the model to.
                                              Defaults to config.DEVICE.

    Returns:
        torch.nn.Module: The configured PyTorch model.
    """
    if device is None:
        device = config.DEVICE

    # Create the model using timm.
    # When num_classes is provided and differs from the pretrained model's default (usually 1000),
    # timm automatically resets the classifier head (fc layer) to match the new number of classes.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)

    return model
