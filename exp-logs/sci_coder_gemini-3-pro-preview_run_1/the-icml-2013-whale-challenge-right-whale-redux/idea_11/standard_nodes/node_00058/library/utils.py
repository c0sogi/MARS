import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import set_seed, mixup_data


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to library.config.set_seed.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def calculate_auc(y_true, y_score):
    """
    Calculates the Area Under the ROC Curve.
    Handles cases where y_true contains only one class by returning 0.5.

    Args:
        y_true (array-like): Ground truth labels.
        y_score (array-like): Predicted probabilities.

    Returns:
        float: The AUC score.
    """
    # Ensure inputs are numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # This exception is raised if y_true has only one class (e.g., all 0s or all 1s).
        # In such cases, AUC is undefined, so we return a neutral score.
        return 0.5


def Mixup(x, y, alpha=0.4):
    """
    Helper function to generate mixed inputs and targets for training regularization.
    Delegates to library.config.mixup_data.

    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter.

    Returns:
        tuple: (mixed_x, y_a, y_b, lam) where:
            - mixed_x: The mixed input tensor.
            - y_a: The first set of labels.
            - y_b: The second set of labels.
            - lam: The mixing coefficient.
    """
    return mixup_data(x, y, alpha)
