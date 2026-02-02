import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Useful for tracking losses and metrics during training epochs.
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


def kl_divergence_score(y_pred, y_true, epsilon=1e-15):
    """
    Computes the Kullback-Leibler Divergence between predicted probabilities and target probabilities.
    Metric = sum(y_true * log(y_true / y_pred))

    Args:
        y_pred: Predicted probabilities (N, C). Can be torch.Tensor or np.ndarray.
        y_true: Target probabilities (N, C). Can be torch.Tensor or np.ndarray.
        epsilon: Small constant to avoid log(0).

    Returns:
        float: The mean KL divergence over the batch.
    """
    # Convert Tensors to NumPy if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Clip predictions to avoid undefined log(0)
    # We do not clip y_true because 0 * log(0) is handled by the mask below
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Initialize result array
    kl_matrix = np.zeros_like(y_true)

    # Only compute terms where y_true > 0 to avoid 0 * log(0) / undefined
    # KL(P||Q) = sum(P * (log(P) - log(Q)))
    mask = y_true > 0

    kl_matrix[mask] = y_true[mask] * (np.log(y_true[mask]) - np.log(y_pred[mask]))

    # Sum over classes (axis 1) to get KL per sample
    sample_kl = np.sum(kl_matrix, axis=1)

    # Return mean over the batch
    return np.mean(sample_kl)


def softmax_and_normalize(logits):
    """
    Applies softmax to logits and explicitly renormalizes to ensure rows sum to 1.0.

    Args:
        logits: Input tensor or array of shape (N, C).

    Returns:
        np.ndarray: Probabilities of shape (N, C) summing to 1.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Apply softmax
    probs = F.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()

    # Explicitly normalize to handle potential floating point drift
    row_sums = probs.sum(axis=1, keepdims=True)

    # Safety check for zero sums (unlikely with softmax)
    row_sums[row_sums == 0] = 1.0

    probs = probs / row_sums

    return probs


def normalize_probabilities(probs):
    """
    Renormalizes an array of probabilities to ensure they sum to 1.0.
    Useful for post-processing ensembles.

    Args:
        probs: Array of probabilities (N, C).

    Returns:
        np.ndarray: Normalized probabilities.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return probs / row_sums
