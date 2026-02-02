import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like or torch.Tensor): True binary labels.
        y_scores (array-like or torch.Tensor): Target scores (probability estimates).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Check if y_true contains only one class (edge case in small batches or debug runs)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_scores)


def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves the model checkpoint including optimizer state and epoch.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current epoch.
        path (str): The file path to save the checkpoint.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    torch.save(checkpoint, path)
