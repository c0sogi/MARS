import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Instantiates a PyTorch model based on the provided architecture name.

    This function loads a backbone from torchvision (ResNet34 or ResNeXt50_32x4d),
    optionally loads pretrained ImageNet weights, and replaces the final
    fully connected layer to match the number of target classes.

    Args:
        model_name (str): The name of the architecture to use.
                          Must be one of ['resnet34', 'resnext50_32x4d'].
        num_classes (int): The number of output classes for the final layer.
        pretrained (bool): If True, loads the 'DEFAULT' pretrained weights
                           (typically ImageNet). Defaults to True.

    Returns:
        nn.Module: The modified PyTorch model ready for training/inference.

    Raises:
        ValueError: If an unsupported model_name is provided.
    """
    # Determine weights parameter based on pretrained flag
    # "DEFAULT" loads the best available weights for the specific model version
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=weights)
    else:
        raise ValueError(
            f"Model architecture '{model_name}' is not supported. "
            "Expected 'resnet34' or 'resnext50_32x4d'."
        )

    # Modify the final fully connected layer (fc)
    # Both ResNet and ResNeXt in torchvision use 'fc' as the final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
