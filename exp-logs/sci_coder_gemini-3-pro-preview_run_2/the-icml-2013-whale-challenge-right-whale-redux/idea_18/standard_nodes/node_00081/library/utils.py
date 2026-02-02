import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_pred: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate AUC
    # In rare cases (e.g., small batches), only one class might be present.
    # sklearn raises a ValueError in this case.
    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        score = 0.5

    return score
