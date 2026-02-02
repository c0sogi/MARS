import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def inverse_scale(mean_norm, sigma_norm):
    """
    Transforms normalized predictions back to the original scale (ml).

    Args:
        mean_norm: Normalized FVC mean prediction.
        sigma_norm: Normalized FVC uncertainty prediction.

    Returns:
        mean_abs: Absolute FVC mean in ml.
        sigma_abs: Absolute FVC uncertainty in ml.
    """
    mean_abs = mean_norm * Config.TARGET_STD + Config.TARGET_MEAN
    sigma_abs = sigma_norm * Config.TARGET_STD
    return mean_abs, sigma_abs


def laplace_log_likelihood(y_true, y_pred, sigma, clip_sigma=True, clip_delta=True):
    """
    Calculates the modified Laplace Log Likelihood metric.
    Metric values are negative and higher is better.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC (ml).
        y_pred: Predicted FVC (ml).
        sigma: Predicted confidence (ml).
        clip_sigma: Whether to clip sigma at 70ml (standard for metric).
        clip_delta: Whether to clip delta at 1000ml (standard for metric).

    Returns:
        The average metric score over the batch.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Absolute difference
    delta = np.abs(y_true - y_pred)

    # Clipping
    if clip_delta:
        delta = np.minimum(delta, 1000.0)

    sigma_clipped = sigma
    if clip_sigma:
        sigma_clipped = np.maximum(sigma, 70.0)

    # Metric calculation
    metric = -(np.sqrt(2) * delta / sigma_clipped) - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


class LaplaceNLLLoss(nn.Module):
    """
    Differentiable Negative Log Likelihood loss for the Laplace distribution.
    Designed to align with the competition metric by operating in the unscaled (ml) space.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_mean_norm, pred_sigma_norm, target_norm):
        """
        Args:
            pred_mean_norm: Model output for mean (normalized).
            pred_sigma_norm: Model output for uncertainty (normalized).
            target_norm: Ground truth target (normalized).

        Returns:
            Loss value (scalar).
        """
        # 1. Inverse scale to get back to ML units
        # We unscale both predictions and targets to compute loss in the metric space.
        # This ensures the '70ml' floor and logarithmic penalties are applied correctly.
        pred_mean = pred_mean_norm * Config.TARGET_STD + Config.TARGET_MEAN
        pred_sigma = pred_sigma_norm * Config.TARGET_STD
        target = target_norm * Config.TARGET_STD + Config.TARGET_MEAN

        # 2. Apply architectural constraints
        # Ensure sigma is at least 70ml to match metric definition and prevent collapse
        sigma_clipped = torch.clamp(pred_sigma, min=Config.SIGMA_FLOOR)

        # 3. Calculate NLL
        # Note: We do NOT clip the delta (error) at 1000ml during training.
        # Clipping delta would kill gradients for large errors, hindering convergence.
        delta = torch.abs(target - pred_mean)

        loss = (
            torch.sqrt(torch.tensor(2.0).to(delta.device)) * delta
        ) / sigma_clipped + torch.log(
            torch.sqrt(torch.tensor(2.0).to(delta.device)) * sigma_clipped
        )

        return torch.mean(loss)
