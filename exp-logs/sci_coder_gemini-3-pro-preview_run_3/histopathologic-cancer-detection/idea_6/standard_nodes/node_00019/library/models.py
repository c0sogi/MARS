import torch
import torch.nn as nn
import timm


def get_model(model_name, pretrained=True):
    """
    Model factory that creates a model using the timm library and configures
    the final classification head for binary classification.

    Args:
        model_name (str): The name of the architecture to create.
                          Supported: 'convnext_tiny', 'tf_efficientnetv2_s'.
        pretrained (bool): If True, loads weights pretrained on ImageNet.

    Returns:
        torch.nn.Module: The PyTorch model with a single output unit (logit).
    """

    # Validate supported models based on the task description
    supported_models = ["convnext_tiny", "tf_efficientnetv2_s"]
    if model_name not in supported_models:
        raise ValueError(
            f"Model '{model_name}' is not supported. Choose from {supported_models}."
        )

    try:
        # Create the model using timm.
        # num_classes=1 automatically replaces the final fully connected layer
        # with a new layer having 1 output node (logit).
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

        return model

    except Exception as e:
        raise RuntimeError(f"Error creating model '{model_name}': {e}")
