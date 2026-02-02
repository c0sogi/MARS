import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("architecture")


def get_model(model_key, pretrained=True):
    """
    Instantiates a model based on the provided key from Config.MODELS.

    Uses the 'timm' library to load the backbone with specified pre-trained weights
    and replaces the classification head with a linear layer outputting a single logit
    (num_classes=1) for binary classification.

    Args:
        model_key (str): The key identifying the model architecture (e.g., 'resnet50', 'convnext_small').
        pretrained (bool): Whether to load pre-trained ImageNet weights. Defaults to True.

    Returns:
        torch.nn.Module: The PyTorch model ready for training or inference.
    """
    if model_key not in Config.MODELS:
        raise ValueError(
            f"Model key '{model_key}' not found in Config.MODELS. Available keys: {list(Config.MODELS.keys())}"
        )

    model_cfg = Config.MODELS[model_key]
    timm_model_name = model_cfg["model_name"]

    logger.info(f"Initializing model: {timm_model_name} (Pretrained: {pretrained})")

    try:
        # timm.create_model handles loading the backbone weights and initializing
        # the new head for num_classes=1 automatically.
        model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=1)
    except Exception as e:
        logger.error(f"Failed to create model '{timm_model_name}': {e}")
        raise e

    return model
