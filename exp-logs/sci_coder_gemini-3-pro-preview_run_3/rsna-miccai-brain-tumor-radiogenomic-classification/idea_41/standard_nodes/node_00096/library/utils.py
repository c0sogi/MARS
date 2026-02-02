import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything, DEVICE


def get_device():
    """
    Returns the PyTorch device object based on the configuration constant.

    Returns:
        torch.device: The device (CPU or CUDA) to be used for computation.
    """
    return torch.device(DEVICE)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC) between ground truth and predictions.

    This function handles inputs as either NumPy arrays or PyTorch tensors. It also
    safeguards against cases where the ground truth contains only one class (which
    would raise a ValueError in sklearn), returning 0.5 in such instances.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities for the positive class.

    Returns:
        float: The ROC AUC score.
    """
    # Detach and convert to numpy if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure correct shape for metric calculation
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Check for single-class edge case in the current batch/set
    # ROC AUC is undefined if there is only one class in y_true
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
