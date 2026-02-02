import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies MixUp augmentation to the input batch.
    Returns mixed inputs, pairs of targets, and lambda.

    Args:
        x (torch.Tensor): Input batch (images/spectrograms).
        y (torch.Tensor): Target batch (labels).
        alpha (float): Parameter for the Beta distribution.
        device (str): Device to perform computations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixing coefficient.
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
    Computes the loss for MixUp augmented data.

    Args:
        criterion (callable): The loss function (e.g., KLDivLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def kl_divergence_score(y_pred, y_true, epsilon=1e-6):
    """
    Computes the Kullback-Leibler Divergence between predictions and targets.
    This serves as the evaluation metric.

    Note: PyTorch's F.kl_div expects input to be log-probabilities.
    Since the model outputs probabilities (Softmax), we take the log here.

    Args:
        y_pred (torch.Tensor): Predicted probabilities (batch_size, num_classes).
        y_true (torch.Tensor): Ground truth probabilities (batch_size, num_classes).
        epsilon (float): Small value to prevent log(0).

    Returns:
        float: The average KL divergence over the batch.
    """
    # Clip predictions to avoid log(0) errors, though usually handled by log_softmax if used directly
    y_pred = torch.clamp(y_pred, min=epsilon, max=1.0)

    # Calculate KL Divergence
    # F.kl_div(input, target): input should be log-probs, target should be probs
    # reduction='batchmean' mathematically aligns with the definition of KL divergence
    loss = F.kl_div(torch.log(y_pred), y_true, reduction="batchmean")

    return loss.item()
