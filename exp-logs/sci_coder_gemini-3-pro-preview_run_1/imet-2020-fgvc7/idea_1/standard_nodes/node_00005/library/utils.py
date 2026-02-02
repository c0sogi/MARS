import os
import random
import numpy as np
import torch
import warnings
from sklearn.metrics import f1_score

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_micro_f1(preds, targets, threshold: float = 0.5):
    """
    Calculates the Micro F1 score for multi-label classification.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, C).
        targets (np.ndarray or torch.Tensor): Ground truth binary labels of shape (N, C).
        threshold (float): Probability threshold to convert predictions to binary (0 or 1).

    Returns:
        float: The micro-averaged F1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Apply threshold to predictions to get binary outputs
    preds_binary = (preds > threshold).astype(int)
    targets_binary = targets.astype(int)

    # Calculate Micro F1 score
    # zero_division=0 ensures no error if there are no positive labels/predictions
    score = f1_score(targets_binary, preds_binary, average="micro", zero_division=0)

    return score
