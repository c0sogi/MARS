import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import GLUBlock


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def custom_weight_init(model):
    """
    Traverses the model to apply specific initialization schemes:
    - Positional Embeddings: Low-variance Normal (std=0.02).
    - GLU Linear Layers: Xavier Uniform (avoiding Kaiming).
    - Transformer Attention: Xavier Uniform.

    Args:
        model (nn.Module): The PyTorch model to initialize.
    """
    # 1. Positional Embeddings
    # Check if the model has a 'pos_embed' parameter (specific to StochasticDepthHybridNet)
    if hasattr(model, "pos_embed") and isinstance(model.pos_embed, nn.Parameter):
        nn.init.normal_(model.pos_embed, mean=0.0, std=0.02)

    # 2. Traverse Modules
    for name, module in model.named_modules():
        # GLU Linear Layers
        if isinstance(module, GLUBlock):
            # Apply Xavier Uniform to fc1
            if hasattr(module, "fc1") and isinstance(module.fc1, nn.Linear):
                nn.init.xavier_uniform_(module.fc1.weight)
                if module.fc1.bias is not None:
                    nn.init.zeros_(module.fc1.bias)

            # Apply Xavier Uniform to fc2
            if hasattr(module, "fc2") and isinstance(module.fc2, nn.Linear):
                nn.init.xavier_uniform_(module.fc2.weight)
                if module.fc2.bias is not None:
                    nn.init.zeros_(module.fc2.bias)

        # Transformer Attention
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize input projections (q, k, v)
            # PyTorch MultiheadAttention packs these into in_proj_weight usually
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            else:
                # Handle cases where weights are separated
                if module.q_proj_weight is not None:
                    nn.init.xavier_uniform_(module.q_proj_weight)
                if module.k_proj_weight is not None:
                    nn.init.xavier_uniform_(module.k_proj_weight)
                if module.v_proj_weight is not None:
                    nn.init.xavier_uniform_(module.v_proj_weight)

            # Initialize output projection
            if module.out_proj is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
