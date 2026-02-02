import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CUDA backend for deterministic execution.

    Args:
        seed (int): The seed value to use.
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


def custom_weight_init(m):
    """
    Applies specific weight initialization schemes based on the layer type.
    - nn.Linear: Kaiming Uniform (He initialization).
    - nn.MultiheadAttention: Xavier Uniform (Glorot initialization).

    Args:
        m (nn.Module): The module to initialize.
    """
    # Note: We rely on PyTorch defaults for nn.Linear to avoid ReLU-specific init on GLU layers (Cite solution_lesson_node_00082).

    # Initialize Transformer Attention layers with Xavier Uniform
    if isinstance(m, nn.MultiheadAttention):
        # Initialize input projection weights (Q, K, V)
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)

        # Initialize output projection weights.
        # Note: m.out_proj is an nn.Linear submodule. Since .apply() visits children
        # before parents, it will have been initialized with Kaiming above.
        # We overwrite it here to strictly enforce Xavier for all attention components.
        if m.out_proj is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
