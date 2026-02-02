import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for cuDNN.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard score between two strings.

    Args:
        str1 (str): The first string (e.g., ground truth).
        str2 (str): The second string (e.g., prediction).

    Returns:
        float: The Jaccard similarity score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    denominator = len(a) + len(b) - len(c)
    if denominator == 0:
        return 0.0

    return float(len(c)) / denominator


def get_optimizer_params(model, config):
    """
    Configures the optimizer parameters with Differential Learning Rates (DLR)
    and Global Weight Decay as per the strategy.

    Strategy:
    - Backbone parameters get a lower learning rate (config.lr_backbone).
    - Task-specific head parameters get a higher learning rate (config.lr_head).
    - Weight decay is applied to ALL parameters (including biases and LayerNorms)
      to act as a strong regularizer for the small dataset.

    Args:
        model (torch.nn.Module): The model instance.
        config (Config): Configuration object containing learning rates and weight decay.

    Returns:
        list: A list of parameter group dictionaries for the optimizer.
    """
    # Identify the backbone module using explicit references
    # XLM-Roberta models typically store the transformer in 'roberta'
    if hasattr(model, "roberta"):
        backbone = model.roberta
    elif hasattr(model, "backbone"):
        backbone = model.backbone
    elif hasattr(model, "transformer"):
        backbone = model.transformer
    else:
        # Fallback to standard XLM-R naming if generic attributes aren't found
        # This assumes the model follows standard HuggingFace naming conventions
        try:
            backbone = model.roberta
        except AttributeError:
            raise AttributeError(
                "Could not identify backbone module (expected 'roberta', 'backbone', or 'transformer')."
            )

    # Get the object IDs of the backbone parameters to distinguish them
    backbone_params_ids = set(map(id, backbone.parameters()))

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if id(param) in backbone_params_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Apply Global Weight Decay to ALL parameters (including bias/LayerNorm)
    # This is a specific strategy for this low-resource task to prevent overfitting.
    optimizer_parameters = [
        {
            "params": backbone_params,
            "lr": config.lr_backbone,
            "weight_decay": config.weight_decay,
        },
        {
            "params": head_params,
            "lr": config.lr_head,
            "weight_decay": config.weight_decay,
        },
    ]

    return optimizer_parameters
