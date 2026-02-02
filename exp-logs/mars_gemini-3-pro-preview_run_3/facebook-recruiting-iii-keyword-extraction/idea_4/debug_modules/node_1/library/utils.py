import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to apply.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(y_pred, y_true, threshold=0.3):
    """
    Computes the Mean F1-Score (micro-averaged) for multi-label classification tasks.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities of shape (N, num_classes).
        y_true (torch.Tensor or np.ndarray): Ground truth binary labels of shape (N, num_classes).
        threshold (float): Probability threshold to convert predictions to binary labels.
                           Defaults to 0.3.

    Returns:
        float: The micro-averaged F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = y_true.astype(int)

    # Calculate Micro-Averaged F1 Score
    # 'micro': Calculate metrics globally by counting the total true positives,
    # false negatives and false positives.
    score = f1_score(y_true_binary, y_pred_binary, average="micro")

    return score
