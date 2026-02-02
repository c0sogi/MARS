import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss.
    Metric formula: - (sqrt(2) * Delta / Sigma_clipped) - ln(sqrt(2) * Sigma_clipped)
    Loss = -Metric (since we want to maximize the metric).
    """

    def __init__(self, reduction="mean"):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        preds: Tensor of shape [batch_size, 2] containing (FVC_pred, Sigma_pred)
        targets: Tensor of shape [batch_size] or [batch_size, 1] containing FVC_true
        """
        fvc_pred = preds[:, 0]
        sigma_pred = preds[:, 1]

        # Ensure targets have correct shape
        if targets.ndim == 2:
            fvc_true = targets.squeeze(1)
        else:
            fvc_true = targets

        # Clip confidence (sigma) to reflect approximate measurement uncertainty
        sigma_clipped = torch.clamp(sigma_pred, min=Config.MIN_CONFIDENCE)

        # Calculate absolute error (Delta) and clip it
        delta = torch.abs(fvc_true - fvc_pred)
        delta = torch.clamp(delta, max=Config.MAX_ERROR)

        # Calculate Metric
        # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=preds.device))

        metric = -(sqrt_2 * delta / sigma_clipped) - torch.log(sqrt_2 * sigma_clipped)

        # Loss is negative metric (minimizing loss -> maximizing metric)
        loss = -metric

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def calculate_metric(preds, targets):
    """
    Helper function to calculate the actual metric score (higher is better).
    Can handle both PyTorch Tensors and NumPy arrays.

    preds: [N, 2] (FVC, Sigma)
    targets: [N] (True FVC)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    fvc_pred = preds[:, 0]
    sigma_pred = preds[:, 1]
    fvc_true = targets.flatten()

    # Apply clipping as per metric definition
    sigma_clipped = np.maximum(sigma_pred, Config.MIN_CONFIDENCE)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), Config.MAX_ERROR)

    # Calculate metric
    metric = -(np.sqrt(2) * delta / sigma_clipped) - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)
