import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_f1_score(y_true, y_pred, threshold=0.5):
    """
    Calculates the Mean F1-Score (Macro-averaged) for multi-label classification.

    Args:
        y_true: Ground truth labels (N, C). Can be torch.Tensor or numpy.ndarray.
        y_pred: Predicted probabilities (N, C). Can be torch.Tensor or numpy.ndarray.
        threshold: Threshold for converting probabilities to binary labels.

    Returns:
        float: The macro-averaged F1 score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Apply threshold to convert probabilities to binary predictions
    y_pred_bin = (y_pred > threshold).astype(int)
    y_true = y_true.astype(int)

    # Calculate Macro F1
    # zero_division=0 prevents warnings/errors when precision/recall is 0
    return f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
