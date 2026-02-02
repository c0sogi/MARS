import os
import random
import numpy as np
import torch
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms ensure reproducibility but may reduce performance slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer_params(model: torch.nn.Module, weight_decay: float = 1e-2):
    """
    Separates model parameters into two groups for the optimizer:
    1. Parameters to apply weight decay (Linear weights, Embeddings).
    2. Parameters to exclude from weight decay (Biases, LayerNorm parameters).

    Args:
        model (torch.nn.Module): The model to optimize.
        weight_decay (float): The weight decay coefficient for the first group.

    Returns:
        list: A list of dictionaries defining the parameter groups.
    """
    decay_params = []
    no_decay_params = []

    # Iterate through named parameters to classify them
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Heuristic: Biases and Normalization parameters (containing 'norm' or 'bias')
        # should not have weight decay applied.
        # Also exclude embeddings from weight decay to preserve positional/token signal (Cite solution_lesson_node_00098).
        if "bias" in name or "norm" in name.lower() or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return optimizer_grouped_parameters


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return roc_auc_score(y_true, y_pred)
