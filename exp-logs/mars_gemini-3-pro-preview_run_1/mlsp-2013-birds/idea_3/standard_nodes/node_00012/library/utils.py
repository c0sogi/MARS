import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input data batch (e.g., images/spectrograms).
        y (torch.Tensor): Target labels batch.
        alpha (float): Mixup interpolation coefficient (Beta distribution parameter).
        use_cuda (bool): Whether to use CUDA for generating the permutation index.

    Returns:
        mixed_x (torch.Tensor): The mixed input data.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The permuted targets.
        lam (float): The lambda mixing factor sampled from Beta(alpha, alpha).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss for Mixup training.

    Args:
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The permuted targets.
        lam (float): The lambda mixing factor used in mixup_data.

    Returns:
        loss (torch.Tensor): The weighted sum of losses with respect to y_a and y_b.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
