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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient.
        device (torch.device, optional): Device to perform computations on.
                                        If None, uses x.device.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Lambda value used for mixing.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Permuted labels.
        lam (float): Lambda value.

    Returns:
        loss (torch.Tensor): Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_multilabel_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth labels (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities (N, num_classes).

    Returns:
        float: Macro-averaged ROC AUC score.
    """
    try:
        # Standard macro-average ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
        if np.isnan(score):
            raise ValueError("ROC AUC is NaN")
        return score
    except ValueError:
        # Fallback for cases where a class might have only one label in the batch
        # Calculate per class and ignore failing ones
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            try:
                # Check if class has both 0 and 1
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                continue

        if len(scores) > 0:
            return np.mean(scores)
        else:
            return 0.5
