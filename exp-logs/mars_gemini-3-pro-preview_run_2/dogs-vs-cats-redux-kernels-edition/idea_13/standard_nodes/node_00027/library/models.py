import timm
import torch.nn as nn
from library.config import Config


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Creates and returns a model based on the provided architecture name using the timm library.

    This function is designed to instantiate the heterogeneous backbones defined in Config.MODEL_ARCHS
    (ConvNeXt, Swin, EfficientNetV2). By default, it configures the model head for binary
    classification (num_classes=1) to be compatible with BCEWithLogitsLoss and the float targets
    provided by the dataset.

    Args:
        model_name (str): The name of the model architecture to create.
                          Must be a valid model name in the timm registry.
        pretrained (bool): Whether to initialize the model with pretrained weights (e.g., ImageNet).
                           Defaults to True.
        num_classes (int): The number of output units in the final classification head.
                           Defaults to 1 (Logits for Binary Classification).

    Returns:
        nn.Module: The initialized PyTorch model with the specified head configuration.

    Raises:
        RuntimeError: If the model_name is not found in the timm registry or creation fails.
    """
    try:
        # Create the model using timm's factory function.
        # timm automatically handles the replacement of the classification head
        # (e.g., 'fc', 'classifier', 'head') to match num_classes.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to create model '{model_name}'. Ensure the model name is correct and supported by timm. Details: {e}"
        )
