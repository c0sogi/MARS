import torch
import torch.nn as nn
import numpy as np
from library.config import Config, seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training/evaluation.
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
    Custom loss function implementing the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, we minimize the Loss:
        Loss = -Metric
             = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_error = Config.MAX_ERROR
        self.min_sigma = Config.MIN_SIGMA

    def forward(self, pred_slope, pred_conf, base_fvc, time_delta, true_fvc):
        """
        Calculate the loss based on the linear decay model predictions.

        Args:
            pred_slope (Tensor): Predicted rate of decline (alpha). Shape: (B,) or (B, 1)
            pred_conf (Tensor): Predicted confidence/uncertainty (sigma). Shape: (B,) or (B, 1)
            base_fvc (Tensor): The baseline FVC measured at week 0. Shape: (B,) or (B, 1)
            time_delta (Tensor): The relative week number for the prediction. Shape: (B,) or (B, 1)
            true_fvc (Tensor): The ground truth FVC. Shape: (B,) or (B, 1)

        Returns:
            Tensor: The mean loss over the batch.
        """
        # Ensure all inputs are 1D tensors to support broadcasting and avoid shape mismatches
        pred_slope = pred_slope.view(-1)
        pred_conf = pred_conf.view(-1)
        base_fvc = base_fvc.view(-1)
        time_delta = time_delta.view(-1)
        true_fvc = true_fvc.view(-1)

        # 1. Calculate Predicted FVC based on the linear model
        # FVC_pred = Baseline_FVC + (Slope * Time)
        pred_fvc = base_fvc + (pred_slope * time_delta)

        # 2. Calculate Delta (Absolute Error)
        # The metric thresholds errors at 1000ml
        abs_error = torch.abs(true_fvc - pred_fvc)
        delta = torch.clamp(abs_error, max=self.max_error)

        # 3. Calculate Clipped Sigma
        # The metric clips confidence at 70ml
        sigma_clipped = torch.clamp(pred_conf, min=self.min_sigma)

        # 4. Compute Loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        # Term 2: ln(sqrt(2) * sigma)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=pred_slope.device))

        term_1 = (sqrt_2 * delta) / sigma_clipped
        term_2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term_1 + term_2

        return torch.mean(loss)
