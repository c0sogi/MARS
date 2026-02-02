import torch
import torch.nn as nn
import numpy as np
from library.config import Config, seed_everything


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
    Implements the modified Laplace Log Likelihood loss function.

    Metric Definition:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Loss Definition (to minimize):
        Loss = -Metric
             = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.confidence_min = Config.CONFIDENCE_MIN
        self.error_max = Config.ERROR_MAX

    def forward(self, pred_fvc, pred_sigma, target_fvc):
        """
        Calculates the loss for a batch.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values.
            pred_sigma (torch.Tensor): Predicted Confidence (sigma) values.
            target_fvc (torch.Tensor): True FVC values.

        Returns:
            torch.Tensor: The mean loss value to be minimized.
        """
        # Ensure inputs are float
        pred_fvc = pred_fvc.float()
        pred_sigma = pred_sigma.float()
        target_fvc = target_fvc.float()

        # Clip Confidence (sigma)
        # sigma_clipped = max(sigma, 70)
        sigma_clipped = torch.clamp(pred_sigma, min=self.confidence_min)

        # Calculate Absolute Error
        abs_error = torch.abs(target_fvc - pred_fvc)

        # Clip Error (Delta)
        # delta = min(|true - pred|, 1000)
        delta = torch.clamp(abs_error, max=self.error_max)

        # Calculate Loss Terms
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=pred_fvc.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Sum terms to get negative metric (Loss)
        loss = term1 + term2

        return torch.mean(loss)


def calculate_metric(pred_fvc, pred_sigma, target_fvc):
    """
    Calculates the actual competition metric (higher is better).
    Useful for validation scoring.

    Args:
        pred_fvc (Tensor or array): Predicted FVC.
        pred_sigma (Tensor or array): Predicted Confidence.
        target_fvc (Tensor or array): True FVC.

    Returns:
        float: The mean metric score.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(pred_fvc):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if torch.is_tensor(pred_sigma):
        pred_sigma = pred_sigma.detach().cpu().numpy()
    if torch.is_tensor(target_fvc):
        target_fvc = target_fvc.detach().cpu().numpy()

    # Constants
    confidence_min = Config.CONFIDENCE_MIN
    error_max = Config.ERROR_MAX

    # Clip sigma
    sigma_clipped = np.maximum(pred_sigma, confidence_min)

    # Calculate delta
    abs_error = np.abs(target_fvc - pred_fvc)
    delta = np.minimum(abs_error, error_max)

    # Calculate metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
