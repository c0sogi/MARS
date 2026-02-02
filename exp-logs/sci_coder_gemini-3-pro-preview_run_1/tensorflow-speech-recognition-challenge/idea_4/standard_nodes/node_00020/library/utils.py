import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input data batch (e.g., spectrograms).
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution. If > 0, mixup is applied.
        device (torch.device, optional): Device to move the index tensor to.
                                         If None, uses x.device.

    Returns:
        mixed_x (torch.Tensor): The mixed input data.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    # Generate random permutation index
    index = torch.randperm(batch_size).to(device)

    # Mix the inputs
    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Get the corresponding targets
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.

    Args:
        criterion (callable): The loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The shuffled targets.
        lam (float): The mixing coefficient lambda.

    Returns:
        torch.Tensor: The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
