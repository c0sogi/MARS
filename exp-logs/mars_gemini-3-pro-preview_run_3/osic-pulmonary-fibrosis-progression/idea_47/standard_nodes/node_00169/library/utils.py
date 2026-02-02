import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def metric_score(y_true, y_pred_mean, y_pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood metric defined in the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values (ml).
        y_pred_mean (np.array or torch.Tensor): Predicted FVC mean (ml).
        y_pred_sigma (np.array or torch.Tensor): Predicted FVC confidence/std (ml).

    Returns:
        float: The average metric score over the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_mean, torch.Tensor):
        y_pred_mean = y_pred_mean.detach().cpu().numpy()
    if isinstance(y_pred_sigma, torch.Tensor):
        y_pred_sigma = y_pred_sigma.detach().cpu().numpy()

    # 1. Clip the confidence values (sigma) at 70 ml
    sigma_clipped = np.maximum(y_pred_sigma, 70)

    # 2. Calculate the absolute error (delta), clipped at 1000 ml
    delta = np.minimum(np.abs(y_true - y_pred_mean), 1000)

    # 3. Compute the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


class InverseScaler:
    """
    Utility to transform standardized model outputs back to the original scale (ml).
    Uses the global training statistics defined in Config.
    """

    def __init__(self):
        # Load global stats using Config's caching mechanism
        self.mean, self.std = Config.get_target_stats()

    def __call__(self, pred_mean_norm, pred_sigma_norm):
        """
        Inverse transforms the normalized predictions.

        Args:
            pred_mean_norm (np.array or torch.Tensor): Standardized predicted mean.
            pred_sigma_norm (np.array or torch.Tensor): Standardized predicted sigma.

        Returns:
            tuple: (pred_mean_orig, pred_sigma_orig) in ml.
        """
        # Handle Tensor vs Numpy to ensure operations are on the correct device/type
        is_tensor = torch.is_tensor(pred_mean_norm)

        if is_tensor:
            mean_global = torch.tensor(
                self.mean, device=pred_mean_norm.device, dtype=pred_mean_norm.dtype
            )
            std_global = torch.tensor(
                self.std, device=pred_mean_norm.device, dtype=pred_mean_norm.dtype
            )
        else:
            mean_global = self.mean
            std_global = self.std

        # Inverse Z-score for Mean: x = z * std + mean
        pred_mean_orig = pred_mean_norm * std_global + mean_global

        # Inverse Scale for Sigma: sigma = sigma_norm * std
        # Note: Uncertainty is a scaling factor, so we do not add the mean shift.
        pred_sigma_orig = pred_sigma_norm * std_global

        return pred_mean_orig, pred_sigma_orig
