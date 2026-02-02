import timm
import torch.nn as nn


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create model instances using the timm library.
    It configures the model with a single output logit suitable for Binary Cross Entropy.

    Args:
        model_name (str): The specific timm model identifier.
                          Expected values include 'resnet50.a1_in1k' and 'convnext_tiny.fb_in1k'.
        pretrained (bool): Whether to initialize the model with pre-trained ImageNet weights.
                           Defaults to True.

    Returns:
        nn.Module: The PyTorch model with the head modified for binary classification.

    Raises:
        ValueError: If the model_name is not recognized by timm or instantiation fails.
    """
    try:
        # Create the model using timm
        # num_classes=1 replaces the default classifier head with a Linear layer outputting 1 dimension.
        # This single logit is the expected input for BCEWithLogitsLoss.
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    except Exception as e:
        raise ValueError(
            f"Error creating model '{model_name}': {e}. "
            "Please ensure the model name is a valid timm identifier."
        )

    return model
