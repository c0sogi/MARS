import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Computes the average Kullback-Leibler (KL) Divergence between true and predicted probabilities.

    Metric Formula: KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities, shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities, shape (N, C).
        epsilon (float): Small constant to prevent log(0) and division by zero.

    Returns:
        float: The average KL divergence across the batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip values to ensure numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(y_true * (log(y_true) - log(y_pred)))
    # Sum over classes (axis=1), then mean over samples
    kl = np.sum(y_true * (np.log(y_true) - np.log(y_pred)), axis=1)

    return np.mean(kl)
