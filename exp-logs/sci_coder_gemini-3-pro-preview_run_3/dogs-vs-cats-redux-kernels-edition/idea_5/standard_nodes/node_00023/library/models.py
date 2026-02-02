import torch
import torch.nn as nn
import timm


def get_model(
    model_name: str, pretrained: bool = True, num_classes: int = 1
) -> nn.Module:
    """
    Creates and returns a model based on the specified architecture name using the timm library.

    This function handles the instantiation of the backbone (e.g., ResNet-101, ConvNeXt-Small)
    and adapts the classification head for the specific number of target classes (default is 1
    for binary classification with BCEWithLogitsLoss).

    Args:
        model_name (str): The name of the model architecture to create (e.g., 'resnet101.a1_in1k',
                          'convnext_small.fb_in1k'). Must be a valid model name in the timm registry.
        pretrained (bool): Whether to load pretrained weights (usually ImageNet-1k). Defaults to True.
        num_classes (int): The number of output classes. Defaults to 1 for binary classification.

    Returns:
        nn.Module: The PyTorch model with the modified head.
    """
    try:
        # Create the model using timm.
        # passing num_classes tells timm to replace the original classification head
        # (usually 1000 classes for ImageNet) with a new one matching num_classes.
        # This automatically handles different head names (e.g., 'fc' for ResNet, 'head.fc' for ConvNeXt).
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}': {str(e)}")
