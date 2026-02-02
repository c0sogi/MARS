import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like or torch.Tensor): True binary labels.
        y_pred (array-like or torch.Tensor): Target scores, can either be probability
                                             estimates of the positive class or confidence values.

    Returns:
        float: The ROC AUC score.
    """
    # Detach and move to CPU if inputs are torch Tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for Mixup training.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): The predictions from the model.
        y_a (torch.Tensor): The targets for the first image in the mix.
        y_b (torch.Tensor): The targets for the second image in the mix.
        lam (float): The lambda mixing coefficient.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
