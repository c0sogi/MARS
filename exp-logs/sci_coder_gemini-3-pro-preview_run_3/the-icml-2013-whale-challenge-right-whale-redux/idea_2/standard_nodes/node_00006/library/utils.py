import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    Delegates to the existing implementation in Config to avoid code duplication.

    Args:
        seed (int): The seed value to set.
    """
    Config.set_seed(seed)


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input batch of data (e.g., spectrograms).
        y (torch.Tensor): Input batch of targets.
        alpha (float): Hyperparameter for the Beta distribution (alpha > 0).
        device (str or torch.device): The device (cpu/cuda) to create the index tensor on.

    Returns:
        mixed_x (torch.Tensor): The mixup-augmented input data.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The targets corresponding to the shuffled indices.
        lam (float): The mixing coefficient (lambda) sampled from Beta(alpha, alpha).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    # Create mixed inputs
    # Note: x[index, :] handles generic dimensions (Batch, Channels, ...)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Get the pair of targets
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs using the Mixed Losses strategy.
    Computes the weighted sum of the loss against the original targets and
    the loss against the shuffled targets.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The shuffled targets.
        lam (float): The mixing coefficient.

    Returns:
        torch.Tensor: The computed weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC).

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if there is only one class in the targets (e.g., in a small batch)
    # ROC AUC is undefined in this case.
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
