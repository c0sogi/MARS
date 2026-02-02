import timm
import torch.nn as nn
from library.config import Config


def create_model(model_key, pretrained=True):
    """
    Creates a model instance based on the configuration specifications.
    Uses timm to load architectures with specific pre-trained weights.
    Configures the final classification head for binary output.

    Args:
        model_key (str): The key identifying the model architecture in Config.MODEL_SPECS
                         (e.g., 'resnet50', 'convnext_small', 'swin_tiny').
        pretrained (bool): Whether to load pre-trained ImageNet weights. Defaults to True.

    Returns:
        nn.Module: The PyTorch model ready for training or inference.
    """
    if model_key not in Config.MODEL_SPECS:
        available_keys = list(Config.MODEL_SPECS.keys())
        raise ValueError(
            f"Invalid model_key '{model_key}'. Available keys: {available_keys}"
        )

    # Retrieve specific timm model name from config
    # e.g., 'resnet50.a1_in1k' which corresponds to the high-quality V2/A1 weights
    spec = Config.MODEL_SPECS[model_key]
    timm_model_name = spec["model_name"]

    # Instantiate the model using timm
    # num_classes=1 sets the final linear layer to output a single logit
    model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=1)

    return model
