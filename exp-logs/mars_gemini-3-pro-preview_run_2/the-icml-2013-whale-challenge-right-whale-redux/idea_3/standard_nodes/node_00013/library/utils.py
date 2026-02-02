import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like or torch.Tensor): Ground truth binary labels (0 or 1).
        y_pred (array-like or torch.Tensor): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate metric
    # Note: roc_auc_score requires both classes to be present in y_true.
    # If a batch only has one class, this might raise a ValueError.
    # In the context of full validation set evaluation, this should not happen.
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Fallback if only one class is present in the provided set
        return 0.0
