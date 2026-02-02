import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def set_seed(seed=Config.SEED):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient.

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels.
        y_pred (np.array or torch.Tensor): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return matthews_corrcoef(y_true, y_pred)


def optimize_threshold(y_true, y_probs):
    """
    Finds the optimal probability threshold that maximizes the MCC score.

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels.
        y_probs (np.array or torch.Tensor): Predicted probabilities (between 0 and 1).

    Returns:
        float: The threshold value that achieved the highest MCC.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    best_mcc = -1.0
    best_thresh = 0.5

    # Search space: 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    print(f"Optimization Complete. Best MCC: {best_mcc}")
    return best_thresh
