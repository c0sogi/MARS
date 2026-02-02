import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard similarity between two strings.

    Args:
        str1 (str): The prediction string.
        str2 (str): The ground truth string.

    Returns:
        float: The Jaccard score.
    """
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)

    union_len = len(a) + len(b) - len(c)
    if union_len == 0:
        return 0.0

    return float(len(c)) / union_len


def get_optimizer_grouped_parameters(model, config):
    """
    Creates parameter groups for the optimizer with differential learning rates.
    Separates parameters into backbone (lower LR) and heads (higher LR),
    and handles weight decay exclusion for bias/LayerNorm.

    Args:
        model: The model instance (must have a 'backbone' attribute).
        config: Configuration object containing LR_BACKBONE, LR_HEAD, and WEIGHT_DECAY.

    Returns:
        list: A list of dictionaries defining parameter groups for the optimizer.
    """
    # Parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.weight"]

    # Ensure model has the expected structure
    if not hasattr(model, "backbone"):
        raise AttributeError("Model must have a 'backbone' attribute for DLR grouping.")

    # 1. Identify Backbone Parameters
    # We use explicit object identity to ensure robust separation
    backbone_params = list(model.backbone.named_parameters())
    backbone_param_ids = {id(p) for n, p in backbone_params}

    # Group 1: Backbone (Lower LR)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.WEIGHT_DECAY,
            "lr": config.LR_BACKBONE,
        },
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": config.LR_BACKBONE,
        },
    ]

    # 2. Identify Head Parameters
    # Any parameter in the model that is NOT part of the backbone is considered a head parameter.
    head_params_decay = []
    head_params_nodecay = []

    for n, p in model.named_parameters():
        if id(p) not in backbone_param_ids:
            if any(nd in n for nd in no_decay):
                head_params_nodecay.append(p)
            else:
                head_params_decay.append(p)

    # Group 2: Heads (Higher LR)
    optimizer_grouped_parameters.extend(
        [
            {
                "params": head_params_decay,
                "weight_decay": config.WEIGHT_DECAY,
                "lr": config.LR_HEAD,
            },
            {
                "params": head_params_nodecay,
                "weight_decay": 0.0,
                "lr": config.LR_HEAD,
            },
        ]
    )

    return optimizer_grouped_parameters
