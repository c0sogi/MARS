import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input data batch.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter. Defaults to Config.MIXUP_ALPHA.
        device (str): Device to perform tensor operations on. Defaults to Config.DEVICE.

    Returns:
        mixed_x (torch.Tensor): The mixed input tensor.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The shuffled targets.
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
    Calculates the loss for mixed inputs using the mixing coefficient.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The shuffled targets.
        lam (float): The mixing coefficient.

    Returns:
        loss (torch.Tensor): The computed weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_score(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC).
    Handles edge cases where only one class is present in the target labels to prevent errors.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The ROC AUC score, or 0.5 if only one class is present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check if there are at least two classes to calculate ROC AUC
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
