import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the implementation in Config.set_seed to avoid duplication.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def compute_mcc(y_true, y_pred):
    """
    Calculates the Matthews Correlation Coefficient (MCC) between ground truth and predictions.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        y_true (Union[np.ndarray, torch.Tensor]): Ground truth binary labels.
        y_pred (Union[np.ndarray, torch.Tensor]): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are treated as binary integers
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc
