import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def kaiming_init_weights(module):
    """
    Applies Kaiming (He) Uniform initialization to Linear layers, specifically
    targeting architectures with GLU/GELU activations where Xavier initialization
    is suboptimal.

    Also initializes normalization layers (BatchNorm, LayerNorm) to identity.

    Args:
        module (nn.Module): The PyTorch module or model to initialize.
    """
    if isinstance(module, nn.Linear):
        # Kaiming Uniform is preferred for GELU/GLU variants
        nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        # Initialize normalization layers to identity
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Detach tensors if necessary and move to cpu/numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
