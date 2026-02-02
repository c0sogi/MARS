import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function.

    Metric Definition:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    This module returns the negative of the metric (Loss) to be minimized.
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_error = Config.MAX_ERROR
        self.min_confidence = Config.MIN_CONFIDENCE
        # Pre-compute sqrt(2)
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Shape (Batch, 2).
                                  Column 0 is FVC_pred, Column 1 is Confidence (sigma).
            targets (torch.Tensor): Shape (Batch) or (Batch, 1). True FVC values.

        Returns:
            loss (torch.Tensor): Scalar tensor containing the mean loss.
        """
        # Ensure correct shapes
        fvc_pred = preds[:, 0].view(-1)
        confidence_pred = preds[:, 1].view(-1)
        fvc_true = targets.view(-1)

        # 1. Clip Confidence (sigma)
        # "confidence values are clipped at 70 ml"
        sigma_clipped = torch.clamp(confidence_pred, min=self.min_confidence)

        # 2. Calculate Absolute Error
        abs_error = torch.abs(fvc_true - fvc_pred)

        # 3. Threshold Error (Delta)
        # "error is thresholded at 1000 ml"
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute Loss terms
        # Loss = -Metric
        # Metric = - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
        # Loss   =   (sqrt(2) * delta / sigma) + ln(sqrt(2) * sigma)

        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


def compute_metric_score(preds, targets):
    """
    Computes the actual competition metric score (Higher is better).
    Useful for validation logging.

    Args:
        preds (torch.Tensor): Predicted FVC and Confidence.
        targets (torch.Tensor): True FVC.

    Returns:
        float: The average metric score (negative value).
    """
    loss_fn = LaplaceLogLikelihoodLoss()
    with torch.no_grad():
        # The class returns Loss (which is -Metric)
        loss = loss_fn(preds, targets)
        # Return Metric = -Loss
        return -loss.item()
