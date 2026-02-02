import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility across
    random, numpy, and torch libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_params(model, weight_decay: float = 1e-2):
    """
    Splits model parameters into two groups for AdamW optimizer:
    1. Weights with decay (Linear layers, Embeddings, Attention projections).
    2. Parameters with 0.0 decay (Biases, LayerNorm parameters, and pos_embed).

    Args:
        model (torch.nn.Module): The model to optimize.
        weight_decay (float): The weight decay factor for the first group.

    Returns:
        list: A list of dictionaries defining the parameter groups suitable for
              initialization of a PyTorch optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Specific exclusion for 'pos_embed' to prevent regularization artifacts
        #    on the learned positional encodings.
        if "pos_embed" in name:
            no_decay_params.append(param)

        # 2. General exclusion for Biases and Normalization parameters.
        #    - Biases are typically 1D.
        #    - LayerNorm weights (gamma) and biases (beta) are typically 1D.
        #    Checking ndim < 2 is a robust heuristic for these.
        elif param.ndim < 2:
            no_decay_params.append(param)

        # 3. Fallback explicit check for 'bias' in the name.
        elif "bias" in name:
            no_decay_params.append(param)

        # 4. All other parameters (e.g., Linear weights, Embedding weights) get decay.
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
