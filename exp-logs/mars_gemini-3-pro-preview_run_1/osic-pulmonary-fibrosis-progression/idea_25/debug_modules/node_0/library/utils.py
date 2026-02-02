import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_loss(
    y_true,
    y_pred,
    sigma,
    clip_sigma=Config.Q_SIGMA_THRESHOLD,
    clip_error=Config.ERROR_THRESHOLD,
):
    """
    Calculates the modified Laplace Log Likelihood loss (negative metric).

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    We minimize the Loss = -Metric:
        loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): Ground truth FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted confidence (standard deviation).
        clip_sigma (float): Minimum threshold for sigma (default 70 ml).
        clip_error (float): Maximum threshold for absolute error (default 1000 ml).

    Returns:
        torch.Tensor: The mean loss over the batch.
    """
    # Flatten tensors to ensure shape alignment (Batch_Size,)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    sigma = sigma.view(-1)

    # 1. Clip Confidence (sigma)
    # Metric requirement: max(sigma, 70)
    sigma_clipped = torch.clamp(sigma, min=clip_sigma)

    # 2. Calculate and Clip Absolute Error (delta)
    # Metric requirement: min(|true - pred|, 1000)
    abs_error = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_error, max=clip_error)

    # 3. Calculate Loss terms
    # Constant sqrt(2) on the correct device
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

    # Term 1: (sqrt(2) * delta) / sigma_clipped
    term1 = (sqrt_2 * delta) / sigma_clipped

    # Term 2: ln(sqrt(2) * sigma_clipped)
    term2 = torch.log(sqrt_2 * sigma_clipped)

    # Loss = Term1 + Term2
    loss = term1 + term2

    # Return mean loss over the batch
    return torch.mean(loss)
