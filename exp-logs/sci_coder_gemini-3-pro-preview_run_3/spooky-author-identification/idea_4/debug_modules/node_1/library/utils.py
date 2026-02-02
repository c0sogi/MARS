import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss with specific rescaling and clipping.

    This function mimics the evaluation metric by:
    1. Rescaling probabilities so each row sums to 1.
    2. Clipping probabilities to [1e-15, 1-1e-15].
    3. Calculating log loss.

    Args:
        y_true: Array-like of shape (n_samples,). Ground truth labels.
                Can be strings ('EAP', 'HPL', 'MWS') or integers (0, 1, 2).
        y_pred: Array-like of shape (n_samples, n_classes). Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Rescale probabilities: divide each row by the row sum
    # Use keepdims=True to allow broadcasting
    row_sums = y_pred.sum(axis=1, keepdims=True)

    # Avoid division by zero (though unlikely with softmax)
    row_sums[row_sums == 0] = 1.0
    y_pred_rescaled = y_pred / row_sums

    # Clip probabilities to avoid log(0) and log(1) extremes
    # Range: [1e-15, 1 - 1e-15]
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred_rescaled, eps, 1 - eps)

    # Determine labels to ensure log_loss works even if a batch is missing a class
    # Check if y_true contains strings or numbers
    if y_true.dtype.kind in {"U", "S", "O"}:  # Unicode, String, or Object
        labels = ["EAP", "HPL", "MWS"]
    else:
        # Assuming 0: EAP, 1: HPL, 2: MWS based on alphabetical order
        labels = [0, 1, 2]

    # Calculate Log Loss
    return log_loss(y_true, y_pred_clipped, labels=labels)
