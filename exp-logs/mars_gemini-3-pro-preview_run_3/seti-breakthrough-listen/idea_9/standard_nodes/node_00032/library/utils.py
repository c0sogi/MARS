import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of data.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution.
        device (str): Device to perform calculations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input data.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for Mixup augmentation.

    Args:
        criterion (callable): The loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC-AUC).

    Args:
        y_true (np.ndarray or list): Ground truth binary labels.
        y_pred (np.ndarray or list): Predicted probabilities.

    Returns:
        float: The ROC-AUC score.
    """
    # Ensure inputs are numpy arrays for stability
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle case where only one class is present in y_true to avoid sklearn error
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


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
