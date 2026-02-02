import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_weight_init(m):
    """
    Applies specific initialization schemes based on the layer type:
    - nn.Linear: Kaiming Uniform (for SwiGLU/Backbone).
    - nn.MultiheadAttention: Xavier Uniform (for Attention projections).
    - nn.Embedding: Normal distribution with std=1.0 (Unit Variance).
      If the module has an attribute 'is_pos_emb' set to True, uses std=0.02.
    - nn.LayerNorm/BatchNorm1d: Weight=1.0, Bias=0.0.

    Args:
        m (nn.Module): The module to initialize.
    """
    # Linear Layers (SwiGLU, Dense)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # Embeddings
    elif isinstance(m, nn.Embedding):
        # Default to Unit Variance
        std = 1.0
        # Check for Positional Embedding flag
        if hasattr(m, "is_pos_emb") and m.is_pos_emb:
            std = 0.02

        nn.init.normal_(m.weight, mean=0.0, std=std)

    # Multihead Attention
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if m.out_proj.weight is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)

    # Normalization Layers
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
