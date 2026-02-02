import timm
import torch.nn as nn
from library.config import Config


def create_model(
    model_name: str, pretrained: bool = True, num_classes: int = 1
) -> nn.Module:
    """
    Creates and initializes a model using the timm library.

    This factory function supports the instantiation of various backbones
    (e.g., ResNet, ConvNeXt, Swin Transformer) as defined in the project configuration.
    It handles the loading of pre-trained ImageNet weights and adapts the
    classification head for the specific task (binary classification).

    Args:
        model_name (str): The name of the model architecture to create.
                          Should correspond to a valid model name in the timm library
                          (e.g., 'resnet50.a1_in1k', 'convnext_small.fb_in1k').
        pretrained (bool): Whether to initialize the model with pre-trained weights.
                           Defaults to True.
        num_classes (int): The number of classes for the output layer.
                           Defaults to 1 (for binary classification logits).

    Returns:
        nn.Module: The created PyTorch model with the specified configuration.

    Raises:
        RuntimeError: If the model creation fails via timm.
    """
    try:
        # Create the model using timm's factory function.
        # num_classes=1 ensures the head is replaced with a Linear layer outputting a single logit.
        # pretrained=True downloads and loads the specific weights defined by the model_name tag.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=Config.IMG_SIZE,
        )

        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to create model '{model_name}' using timm. Error: {e}"
        )
