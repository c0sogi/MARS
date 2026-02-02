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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_auc(y_true, y_pred):
    """
    Computes the Macro-Averaged Area Under the ROC Curve.
    Handles cases where specific classes might be absent in the target set
    by skipping them in the average calculation.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N_samples, N_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N_samples, N_classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        # Extract columns for the current class
        class_true = y_true[:, i]
        class_pred = y_pred[:, i]

        # Check if the class exists in the ground truth (needs both 0 and 1)
        if len(np.unique(class_true)) > 1:
            try:
                score = roc_auc_score(class_true, class_pred)
                auc_scores.append(score)
            except ValueError:
                # Fallback if sklearn fails for some reason
                continue
        else:
            # If a class is not present or only present (all 0s or all 1s),
            # we cannot compute AUC for this specific class.
            # We skip it to avoid skewing the metric with arbitrary values.
            continue

    if not auc_scores:
        return 0.5  # Return random guess score if no classes could be evaluated

    return np.mean(auc_scores)
