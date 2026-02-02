import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(
    x, y, alpha: float = Config.MIXUP_ALPHA, device: torch.device = Config.DEVICE
):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input batch (e.g., spectrograms).
        y (torch.Tensor): Target labels.
        alpha (float): Mixup hyperparameter.
        device (torch.device): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Targets for the first permutation.
        y_b (torch.Tensor): Targets for the second permutation.
        lam (float): Lambda value used for mixing.
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
    Computes the loss for Mixup training.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first permutation.
        y_b (torch.Tensor): Targets for the second permutation.
        lam (float): Lambda value used for mixing.

    Returns:
        loss (torch.Tensor): Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_auc(y_true, y_scores):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted probabilities or logits.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays/lists on CPU
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle cases where only one class is present in the batch
        return 0.5
