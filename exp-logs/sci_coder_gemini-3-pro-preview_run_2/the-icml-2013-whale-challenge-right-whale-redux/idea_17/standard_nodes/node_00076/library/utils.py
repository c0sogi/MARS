import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device (cuda or cpu) to be used for computation.
    """
    return torch.device(Config.DEVICE)


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to add.
            n (int): The number of samples associated with this value (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (1 for whale call, 0 for noise).
                Can be a list, numpy array, or torch tensor.
        y_pred: Predicted probabilities for the positive class.
                Can be a list, numpy array, or torch tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert torch tensors to numpy arrays if necessary
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
        # Handle edge cases where only one class is present in the batch
        # This can happen in small batches or highly imbalanced validation sets
        score = 0.5

    return float(score)
