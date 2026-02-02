import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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


def optimize_f1_threshold(y_true, y_pred_probs):
    """
    Performs a grid search over the probability range to find the threshold
    that maximizes the Micro F1 score.

    Args:
        y_true: Ground truth labels (N, num_classes). Can be torch.Tensor or np.ndarray.
        y_pred_probs: Predicted probabilities (N, num_classes). Can be torch.Tensor or np.ndarray.

    Returns:
        best_threshold (float): The threshold value that maximizes F1.
        best_f1 (float): The maximum F1 score achieved.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    best_threshold = 0.5
    best_f1 = -1.0

    # Define search space from Config
    # Using small epsilon to ensure the end value is included if step divides range evenly
    start = Config.threshold_start
    end = Config.threshold_end
    step = Config.threshold_step

    thresholds = np.arange(start, end + 1e-9, step)

    for thresh in thresholds:
        # Apply threshold to get binary predictions
        y_pred_bin = (y_pred_probs > thresh).astype(int)

        # Calculate Micro F1 score
        score = f1_score(y_true, y_pred_bin, average="micro")

        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    return best_threshold, best_f1
