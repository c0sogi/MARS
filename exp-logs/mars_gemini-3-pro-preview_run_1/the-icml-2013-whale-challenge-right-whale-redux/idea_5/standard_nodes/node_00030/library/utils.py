import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of data (e.g., spectrograms).
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient parameter.
                       If alpha > 0, lambda is sampled from Beta(alpha, alpha).
                       If alpha <= 0, lambda is set to 1 (no mixup).
        device (str or torch.device): Device to move the shuffled indices to.

    Returns:
        mixed_x (torch.Tensor): The mixed input data.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Shuffled labels.
        lam (float): The mixing coefficient lambda.
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
    Computes the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): Original labels.
        y_b (torch.Tensor): Shuffled labels.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
