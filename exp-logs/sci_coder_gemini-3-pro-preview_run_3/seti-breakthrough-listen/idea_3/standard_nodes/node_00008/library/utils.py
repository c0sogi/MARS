import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input data batch.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter. Defaults to Config.MIXUP_ALPHA.
        device (str): Device to perform computations on. Defaults to Config.DEVICE.

    Returns:
        mixed_x (torch.Tensor): The mixed input data.
        y_a (torch.Tensor): The target for the first image.
        y_b (torch.Tensor): The target for the second image.
        lam (float): The mixing coefficient.
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
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The target for the first image.
        y_b (torch.Tensor): The target for the second image.
        lam (float): The mixing coefficient.

    Returns:
        loss (torch.Tensor): The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_score(y_true, y_pred):
    """
    Calculates the ROC AUC score.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted probabilities.

    Returns:
        float: Area Under the ROC Curve. Returns 0.5 if only one class is present.
    """
    # Handle edge case where y_true contains only one class (e.g. small batches)
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
