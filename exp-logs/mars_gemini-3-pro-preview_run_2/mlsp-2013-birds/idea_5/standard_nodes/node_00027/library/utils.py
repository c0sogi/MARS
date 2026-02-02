import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Configures cuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use.
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


def calculate_pos_weights(y_train, device="cpu"):
    """
    Computes the pos_weight for BCEWithLogitsLoss based on the ratio of
    negative to positive samples for each class in the training set.

    Args:
        y_train (np.ndarray or torch.Tensor): The training labels of shape (N, C).
        device (str or torch.device): The device to store the weights on.

    Returns:
        torch.Tensor: Weights of shape (C,) on the specified device.
    """
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32)
    elif isinstance(y_train, torch.Tensor):
        y_train = y_train.float()

    num_samples = y_train.shape[0]
    pos_counts = torch.sum(y_train, dim=0)
    neg_counts = num_samples - pos_counts

    # Calculate weights: neg / pos
    # Add a small epsilon to pos_counts to avoid division by zero
    pos_weights = neg_counts / (pos_counts + 1e-6)

    return pos_weights.to(device)


def mixup_data(x, y, alpha=0.4, device="cpu"):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of images/spectrograms.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup hyperparameter for Beta distribution.
        device (str or torch.device): Device for computation.

    Returns:
        mixed_x (torch.Tensor): Mixed inputs.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
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
        criterion (callable): The loss function (e.g., BCEWithLogitsLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
