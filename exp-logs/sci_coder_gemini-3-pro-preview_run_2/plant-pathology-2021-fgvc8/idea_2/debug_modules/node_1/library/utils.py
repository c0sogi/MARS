import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to be used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    # Disable benchmark to prevent non-deterministic algorithm selection
    torch.backends.cudnn.benchmark = False


def get_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes the Mean F1-Score (Macro F1) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities or binary labels of shape (n_samples, n_classes).
        threshold (float): Threshold to convert probabilities to binary labels.
                           Defaults to 0.5.

    Returns:
        float: The Macro F1 Score.
    """
    # Convert probabilities to binary predictions if input is floating point
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)

    # Ensure ground truth is integer
    y_true_binary = y_true.astype(int)

    # Calculate Macro F1 Score
    # average='macro': Calculate metrics for each label, and find their unweighted mean.
    return f1_score(y_true_binary, y_pred_binary, average="macro")
