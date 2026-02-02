import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (can be list, numpy array, or torch tensor).
        y_pred: Predicted probabilities (can be list, numpy array, or torch tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in the batch/set
        score = 0.5

    return score


class Mixup:
    """
    Implements Mixup regularization.

    Reference: mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float = 0.2, device: str = "cuda"):
        """
        Args:
            alpha (float): Parameter for the Beta distribution.
            device (str): Device to perform computations on.
        """
        self.alpha = alpha
        self.device = device

    def __call__(self, x, y):
        """
        Applies Mixup to the input batch.

        Args:
            x (torch.Tensor): Input images.
            y (torch.Tensor): Target labels.

        Returns:
            mixed_x (torch.Tensor): Mixed images.
            y_a (torch.Tensor): Original targets.
            y_b (torch.Tensor): Permuted targets.
            lam (float): Mixing coefficient.
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam
