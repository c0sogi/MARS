import timm
import torch.nn as nn
from library.config import Config


def create_model(
    model_name, num_classes=Config.NUM_CLASSES, pretrained=True, dropout_rate=0.0
):
    """
    Factory function to create models using timm.

    Args:
        model_name (str): Name of the model architecture in timm.
        num_classes (int): Number of classes for the classification head.
        pretrained (bool): If True, loads weights pretrained on ImageNet.
        dropout_rate (float): Dropout rate for the classification head.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Create model with timm
    # drop_rate argument in timm.create_model handles the dropout before the final classifier
    # for most supported models including ConvNeXt and Swin.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate,
    )

    return model
