import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def custom_log_loss(y_true, y_pred, eps=1e-15):
    """
    Computes the multi-class log loss with specific rescaling and clipping requirements.

    The probabilities are first rescaled so that each row sums to 1.
    Then they are clipped to the range [eps, 1-eps].
    Finally, the log loss is computed.

    Args:
        y_true (array-like): True labels (n_samples,) or one-hot encoded (n_samples, n_classes).
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        eps (float): Clipping threshold. Default is 1e-15.

    Returns:
        float: The computed log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # Rescale probabilities: divide each row by its sum
    # Use keepdims=True to ensure broadcasting works correctly
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Avoid division by zero
    y_pred_rescaled = np.divide(
        y_pred, row_sums, out=np.zeros_like(y_pred), where=row_sums != 0
    )

    # Clip probabilities to avoid log(0)
    # Note: sklearn log_loss also applies clipping, but we do it explicitly
    # to match the task description's pipeline exactly before passing to the metric function.
    y_pred_clipped = np.clip(y_pred_rescaled, eps, 1 - eps)

    # Calculate log loss
    # sklearn.metrics.log_loss handles the label format (indicator vs labels)
    # provided the classes match the columns of y_pred.
    return log_loss(y_true, y_pred_clipped, eps=eps)
