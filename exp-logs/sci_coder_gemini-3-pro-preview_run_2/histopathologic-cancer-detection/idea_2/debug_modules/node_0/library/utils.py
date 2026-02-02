import random
import os
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Parameter for the Beta distribution.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): The mixed input images.
        y_a (torch.Tensor): The original labels.
        y_b (torch.Tensor): The shuffled labels.
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
    Calculates the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The original labels.
        y_b (torch.Tensor): The shuffled labels.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
