import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
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
    Useful for tracking losses and metrics during training.
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


def KL_loss(y_pred, y_true):
    """
    Computes the Kullback-Leibler divergence metric between predicted and target probabilities.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, C).
        y_true (np.ndarray or torch.Tensor): Target probabilities of shape (N, C).

    Returns:
        float: The average KL divergence score.
    """
    epsilon = 1e-15

    # Convert tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Clip to avoid numerical instability (log(0))
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(y_true * log(y_true / y_pred))
    # Decomposed: sum(y_true * (log(y_true) - log(y_pred)))
    # Sum over classes (axis=1)
    kl = np.sum(y_true * (np.log(y_true) - np.log(y_pred)), axis=1)

    # Return mean over the batch
    return np.mean(kl)
