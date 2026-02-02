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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=None):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution. Defaults to Config.MIXUP_ALPHA.
        device (torch.device): Device to perform computations on. If None, uses x.device.

    Returns:
        mixed_x (torch.Tensor): The mixed input batch.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): The mixing coefficient lambda.
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
    Computes the loss for Mixup training.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Permuted targets.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_metric(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score.
    Safely handles cases where specific classes may be absent in the provided batch/split.

    Args:
        y_true (np.ndarray or list): Ground truth binary labels (N, NumClasses).
        y_pred (np.ndarray or list): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        num_classes = y_true.shape[1]
        auc_scores = []

        for i in range(num_classes):
            # Check if the class exists in y_true (needs at least one 0 and one 1)
            # If a class is all 0s or all 1s in the validation set, AUC is undefined for that class.
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                auc_scores.append(auc)

        if not auc_scores:
            return 0.0

        return np.mean(auc_scores)
    except Exception:
        # Return 0.0 in case of critical errors (e.g., dimension mismatch)
        return 0.0
