import os
import random
import numpy as np
import torch
import torch.nn as nn


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


class LaplaceLogLikelihood(nn.Module):
    """
    Implements the modified Laplace Log Likelihood metric as a Loss function.

    The competition metric is defined as:
        Metric = - (sqrt(2) * Delta / Sigma_clipped) - ln(sqrt(2) * Sigma_clipped)

    Since we want to maximize the Metric, we minimize the Loss:
        Loss = -Metric = (sqrt(2) * Delta / Sigma_clipped) + ln(sqrt(2) * Sigma_clipped)

    Constraints:
        Delta = min(|FVC_true - FVC_pred|, 1000)
        Sigma_clipped = max(Sigma_pred, 70)
    """

    def __init__(self, reduction="mean"):
        super(LaplaceLogLikelihood, self).__init__()
        self.reduction = reduction
        # Register sqrt(2) as a buffer so it moves with the model to GPU
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, fvc_pred, sigma_pred, fvc_true):
        """
        Args:
            fvc_pred (Tensor): Predicted FVC values.
            sigma_pred (Tensor): Predicted Confidence (std dev) values.
                                 Assumed to be positive (e.g. output of Softplus).
            fvc_true (Tensor): Ground truth FVC values.

        Returns:
            Tensor: The calculated loss.
        """
        # Ensure shapes are compatible to avoid silent broadcasting errors
        if fvc_pred.shape != fvc_true.shape:
            # Attempt to view fvc_true as fvc_pred (e.g. [B] -> [B, 1])
            fvc_true = fvc_true.view_as(fvc_pred)

        if sigma_pred.shape != fvc_pred.shape:
            sigma_pred = sigma_pred.view_as(fvc_pred)

        # Calculate absolute error
        abs_err = torch.abs(fvc_true - fvc_pred)

        # Apply thresholding to error (Delta)
        # "The error is thresholded at 1000 ml to avoid large errors adversely penalizing results"
        delta = torch.clamp(abs_err, max=1000.0)

        # Apply clipping to confidence (Sigma)
        # "confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty"
        sigma_clipped = torch.clamp(sigma_pred, min=70.0)

        # Calculate Loss components
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        term_1 = (self.sqrt_2 * delta) / sigma_clipped
        term_2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term_1 + term_2

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss
