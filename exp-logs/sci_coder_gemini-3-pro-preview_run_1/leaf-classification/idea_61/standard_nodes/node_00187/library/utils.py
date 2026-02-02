import os
import random
import numpy as np
import torch
import json
import hashlib
from library.config import FLOAT_PRECISION


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_log_loss(y_true, y_pred):
    """
    Computes the multi-class log loss according to the competition metric.

    Rules:
    1. Probabilities are rescaled to sum to 1 per row.
    2. Probabilities are clipped to [1e-15, 1 - 1e-15].
    3. Log loss is calculated as -1/N * sum(y_true * log(y_pred)).

    Args:
        y_true: Ground truth labels. Can be:
                - 1D array of class indices (integers).
                - 2D array of one-hot encoded vectors.
        y_pred: Predicted probabilities. 2D array of shape (n_samples, n_classes).

    Returns:
        float: The multi-class log loss.
    """
    # Ensure inputs are numpy arrays with high precision
    y_pred = np.array(y_pred, dtype=FLOAT_PRECISION)
    y_true = np.array(y_true)

    # 1. Rescale probabilities (each row divided by row sum)
    # This handles cases where raw classifier outputs might not sum strictly to 1
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Avoid division by zero by replacing 0 sums with 1 (though unlikely with valid probabilities)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip probabilities to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # 3. Compute Log Loss
    N = y_pred.shape[0]

    if y_true.ndim == 1:
        # Case: y_true are class indices (integers)
        # Use fancy indexing to extract the probability of the true class
        # This avoids creating a large one-hot matrix

        # Ensure indices are integers
        y_true = y_true.astype(int)

        # Extract prob of true class
        # y_pred[i, y_true[i]] for all i
        prob_true = y_pred[np.arange(N), y_true]

        loss = -np.mean(np.log(prob_true))

    else:
        # Case: y_true is one-hot encoded or soft labels
        # Standard formula
        loss = -np.sum(y_true * np.log(y_pred)) / N

    return loss


def get_config_hash(config_dict):
    """
    Generates a unique MD5 hash for a given configuration dictionary.
    Used for caching intermediate data based on feature/pipeline settings.

    Args:
        config_dict (dict): Dictionary containing configuration parameters.

    Returns:
        str: MD5 hash string.
    """
    # Serialize dictionary to JSON string with sorted keys for determinism
    # default=str handles non-serializable types like numpy types or functions by converting to string
    config_str = json.dumps(config_dict, sort_keys=True, default=str)

    # Generate MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()
