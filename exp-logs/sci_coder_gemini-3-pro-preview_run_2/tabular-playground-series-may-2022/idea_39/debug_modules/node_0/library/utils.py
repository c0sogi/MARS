import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_grouped_parameters(
    model,
    weight_decay_group1=Config.WEIGHT_DECAY_GROUP1,
    weight_decay_group2=Config.WEIGHT_DECAY_GROUP2,
):
    """
    Groups model parameters for the optimizer to apply strict decoupled weight decay.

    Group 1: Weights of Linear, Embeddings, Attention projections (Decay applied).
    Group 2: Biases, Norms (LayerNorm/BatchNorm), and pos_embed (No decay).

    Args:
        model (torch.nn.Module): The model to optimize.
        weight_decay_group1 (float): Weight decay for Group 1. Defaults to Config.WEIGHT_DECAY_GROUP1.
        weight_decay_group2 (float): Weight decay for Group 2. Defaults to Config.WEIGHT_DECAY_GROUP2.

    Returns:
        list: A list of dictionaries defining the parameter groups.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Logic for determining No Decay group:
        # 1. "pos_embed" is explicitly excluded from decay as per requirements.
        # 2. Parameters with ndim < 2 are typically Biases (1D) or Normalization weights/scales (1D).
        #    This covers LayerNorm.weight, BatchNorm1d.weight, and all .bias parameters.
        if "pos_embed" in name or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay_group1,
        },
        {
            "params": no_decay_params,
            "weight_decay": weight_decay_group2,
        },
    ]

    return optimizer_grouped_parameters
