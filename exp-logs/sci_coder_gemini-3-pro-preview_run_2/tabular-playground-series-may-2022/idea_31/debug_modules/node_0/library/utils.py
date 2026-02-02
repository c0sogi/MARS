import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(module):
    """
    Initialize weights with specific strategies:
    - Positional Embeddings: Normal (mean=0, std=0.02) to break symmetry.
    - Linear (SwiGLU/Backbone): Kaiming Uniform (a=sqrt(5)).
    - Embedding (Transformer): Xavier Uniform.
    - LayerNorm: Ones (weight), Zeros (bias).
    """
    # Handle Learnable Positional Embeddings
    # We check for the attribute 'pos_embed' which is specific to the PostNormTransformer
    if hasattr(module, "pos_embed") and isinstance(module.pos_embed, nn.Parameter):
        nn.init.normal_(module.pos_embed, mean=0.0, std=0.02)

    # Handle Standard Layers
    if isinstance(module, nn.Linear):
        # Kaiming Uniform for SwiGLU blocks and other linear projections
        nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Xavier Uniform for Transformer Embeddings
        nn.init.xavier_uniform_(module.weight)

    elif isinstance(module, nn.LayerNorm):
        # Standard LayerNorm initialization
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
