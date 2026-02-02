import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoader to ensure
    each worker has a different random seed.

    Args:
        worker_id (int): The ID of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_metric(y_true, y_pred):
    """
    Computes the macro-averaged ROC AUC score for multi-label classification.
    Robustly handles cases where a class might be missing from the ground truth
    in a specific batch or fold.

    Args:
        y_true (np.ndarray): Ground truth binary labels (N_samples, N_classes).
        y_pred (np.ndarray): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The mean ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n_classes = y_true.shape[1]
    scores = []

    for i in range(n_classes):
        # Check if both classes (0 and 1) are present in the ground truth column
        # roc_auc_score requires both positive and negative samples
        if len(np.unique(y_true[:, i])) == 2:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                # In case of any other calculation error, skip this class
                pass
        else:
            # If a class is constant in y_true (all 0s or all 1s),
            # AUC is undefined. We skip it for the mean calculation.
            pass

    if len(scores) == 0:
        return 0.5  # Return random guessing score if no classes are valid

    return np.mean(scores)
