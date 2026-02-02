import os
import random
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def compute_log_loss(y_true, y_pred, classes):
    """
    Computes the multi-class log loss with specific clipping and normalization.

    The metric requires that predicted probabilities are rescaled so each row sums to 1,
    and then clipped to the range [1e-15, 1-1e-15] to avoid log(0).

    Args:
        y_true (array-like): Ground truth labels (1D array of strings or integers).
        y_pred (array-like): Predicted probabilities (2D array, shape [n_samples, n_classes]).
        classes (list): List of all unique class names corresponding to the columns of y_pred.
                        This is required to correctly map y_true labels to the probability matrix.

    Returns:
        float: The computed log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred, dtype=np.float64)
    y_true = np.array(y_true)

    # 1. Rescale: Divide each row by its sum
    row_sums = y_pred.sum(axis=1)
    # Avoid division by zero for rows that might sum to 0 (safety check)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Encode y_true to One-Hot Matrix
    # We use OneHotEncoder with explicit categories to ensure columns align with y_pred
    # and to handle cases where y_true might not contain all classes in the validation batch.
    ohe = OneHotEncoder(
        categories=[classes], sparse_output=False, handle_unknown="ignore"
    )

    # Reshape y_true to 2D for OneHotEncoder
    y_true_reshaped = y_true.reshape(-1, 1)
    y_true_one_hot = ohe.fit_transform(y_true_reshaped)

    # 4. Compute Log Loss
    # Loss = - (1/N) * sum( sum( y_true_ij * log(y_pred_ij) ) )
    # Since y_true is one-hot, this sums log(p) for the true class of each sample.

    # Element-wise multiplication and sum
    # We sum over classes (axis 1) then average over samples (axis 0)
    log_likelihood = np.sum(y_true_one_hot * np.log(y_pred))
    loss = -log_likelihood / len(y_true)

    return loss
