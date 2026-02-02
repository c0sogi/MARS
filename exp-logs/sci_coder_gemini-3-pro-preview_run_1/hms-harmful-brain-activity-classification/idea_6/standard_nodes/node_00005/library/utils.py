import os
import random
import numpy as np
import torch
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kl_divergence_score(y_true, y_pred, epsilon: float = 1e-15):
    """
    Calculates the Kullback-Leibler (KL) Divergence between the predicted probability
    and the observed target, as defined by the competition metric.

    Metric = (1/N) * sum_i ( sum_j ( P_ij * log(P_ij / Q_ij) ) )
    Where P is y_true and Q is y_pred.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, C).
        epsilon (float): Small value to prevent log(0).

    Returns:
        float: The mean KL divergence score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    # We do not clip y_true because 0 * log(0) is handled mathematically as 0
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate P * log(P)
    # Handle the case where y_true is 0. lim(x->0) x*log(x) = 0
    # We use np.where to safely compute log only where y_true > 0
    term_true = np.where(y_true > 0, y_true * np.log(y_true), 0.0)

    # Calculate P * log(Q)
    term_pred = y_true * np.log(y_pred)

    # KL = sum(P * log(P) - P * log(Q)) = sum(term_true - term_pred)
    # Sum over classes (axis 1)
    kl_per_sample = np.sum(term_true - term_pred, axis=1)

    # Mean over samples
    return np.mean(kl_per_sample)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training.
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
