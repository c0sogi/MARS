import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function ensures that the results are deterministic by fixing the seed
    for various random number generators and configuring CuDNN settings.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    # Note: This might negatively impact performance but is necessary for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC) for binary classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score.
    """
    try:
        # Check if we have both classes in y_true to avoid ValueError
        if len(np.unique(y_true)) < 2:
            # If only one class is present, AUC is undefined/not meaningful in this context.
            # Returning 0.5 as a neutral baseline or handling as error depending on preference.
            # Here we print a warning and return 0.5.
            print(
                "Warning: Only one class present in y_true. ROC AUC score is not defined."
            )
            return 0.5

        score = roc_auc_score(y_true, y_pred)
        return score
    except ValueError as e:
        print(f"Error computing ROC AUC: {e}")
        return 0.0
