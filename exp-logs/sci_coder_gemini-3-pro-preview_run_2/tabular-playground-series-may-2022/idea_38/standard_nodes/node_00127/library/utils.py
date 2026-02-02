import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Also configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.
    Handles inputs that are either numpy arrays or torch tensors.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)


def get_optimizer_params(model, weight_decay: float):
    """
    Constructs parameter groups for the optimizer to implement decoupled weight decay.

    Group 1 (Decay): Weights of Linear, Embedding, Conv layers, Attention projections.
    Group 2 (No Decay): Biases, LayerNorm/BatchNorm parameters, Positional Embeddings.

    Args:
        model: The PyTorch model.
        weight_decay (float): The weight decay coefficient for the decay group.

    Returns:
        list: A list of dictionaries defining the parameter groups.
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Identify parameters that should not be decayed:
        # 1. Positional Embeddings (explicit name check)
        # 2. Biases (explicit name check or 1D)
        # 3. Normalization parameters (explicit name check 'norm' or 1D)
        if "pos_embed" in name or "bias" in name or "norm" in name or param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
