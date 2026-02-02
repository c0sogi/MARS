import os
import random
import numpy as np
import torch


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
    Calculates the Kullback-Leibler Divergence between true and predicted probabilities.
    Metric = sum(P * log(P / Q))

    Args:
        y_true: Array-like or Tensor of ground truth probabilities.
        y_pred: Array-like or Tensor of predicted probabilities.
        epsilon: Small constant to prevent log(0) and division by zero.

    Returns:
        float: The average KL divergence score across samples.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence terms
    # Formula: P(x) * (log(P(x)) - log(Q(x)))
    # We only compute terms where P(x) > 0 to avoid log(0) for P(x)
    mask = y_true > 0
    kl_terms = np.zeros_like(y_true)

    # Compute P * log(P/Q) = P * (log(P) - log(Q))
    kl_terms[mask] = y_true[mask] * (np.log(y_true[mask]) - np.log(y_pred[mask]))

    # Sum over classes (axis 1)
    sample_kl = np.sum(kl_terms, axis=1)

    # Mean over samples (axis 0)
    return np.mean(sample_kl)
