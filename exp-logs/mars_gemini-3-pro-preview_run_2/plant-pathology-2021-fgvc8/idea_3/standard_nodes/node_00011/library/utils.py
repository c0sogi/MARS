import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred, threshold: float = 0.5):
    """
    Calculates the Mean F1-Score (Macro Average) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary labels of shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities or logits of shape (N, num_classes).
        threshold (float): Threshold value to convert probabilities to binary labels.
                           Defaults to 0.5.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # If predictions are floating point (probabilities/logits), apply threshold
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred > threshold).astype(int)

    # Calculate Macro F1 Score
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    return f1_score(y_true, y_pred, average="macro")
