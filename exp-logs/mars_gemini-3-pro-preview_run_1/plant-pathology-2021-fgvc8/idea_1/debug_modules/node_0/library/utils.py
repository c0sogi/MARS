import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_map(classes=None):
    """
    Generates mapping dictionaries between class string labels and integer indices.

    Args:
        classes (list, optional): List of class names. If None, uses Config.CLASSES.

    Returns:
        tuple: (str2int, int2str) dictionaries.
    """
    if classes is None:
        classes = Config.CLASSES

    str2int = {label: idx for idx, label in enumerate(classes)}
    int2str = {idx: label for idx, label in enumerate(classes)}

    return str2int, int2str


def calculate_score(y_true, y_pred, threshold=0.5, average="macro"):
    """
    Computes the Mean F1-Score for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).
        threshold (float): Threshold for converting probabilities to binary labels.
        average (str): Averaging strategy for F1 score. Defaults to 'macro'.

    Returns:
        float: The calculated Mean F1-Score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Binarize predictions based on the threshold
    y_pred_bin = (y_pred > threshold).astype(int)

    # Calculate F1 Score
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    score = f1_score(y_true, y_pred_bin, average=average, zero_division=0)

    return score
