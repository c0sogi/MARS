import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer_params(
    model, weight_decay=Config.WEIGHT_DECAY, learning_rate=Config.LEARNING_RATE
):
    """
    Creates parameter groups for the optimizer with decoupled weight decay.

    Group 1: Parameters to decay (Linear weights, Embeddings, Attention projections).
    Group 2: Parameters to NOT decay (Biases, LayerNorm, Positional Embeddings, LayerScale).
    """
    # Define substrings to identify parameters that should be excluded from decay
    # 'pos_embedding' targets learnable positional encodings
    no_decay = [
        "bias",
        "LayerNorm.weight",
        "norm.weight",
        "norm1.weight",
        "norm2.weight",
        "pos_embedding",
        "pos_embed",
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    return optimizer_grouped_parameters


def init_weights(module):
    """
    Custom initialization function to be applied via model.apply().

    - Linear: Kaiming Uniform (Optimized for SwiGLU backbone).
    - Embedding: Normal with std=1.0 (Optimized for Post-Norm Transformer).
    - MultiheadAttention: Xavier Uniform (Standard Transformer init).
    - LayerNorm: Ones for weights, Zeros for biases.
    """
    if isinstance(module, nn.Linear):
        # Kaiming Uniform is preferred for the SwiGLU backbone
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Explicitly initialize with Unit Variance (std=1.0) for signal propagation
        nn.init.normal_(module.weight, mean=0.0, std=1.0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.MultiheadAttention):
        # Xavier initialization for Transformer attention components
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        # Note: out_proj is a Linear submodule and may be visited separately by apply(),
        # receiving Kaiming init. This is acceptable for the hybrid architecture.
