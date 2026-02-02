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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
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

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Loss formula (to minimize):
        loss = -metric
             = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_fvc_error = Config.MAX_FVC_ERROR
        self.min_confidence = Config.MIN_CONFIDENCE

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Args:
            pred_fvc (torch.Tensor): Predicted FVC values.
            pred_sigma (torch.Tensor): Predicted confidence (sigma) values.
            true_fvc (torch.Tensor): Ground truth FVC values.

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Calculate absolute error
        delta = torch.abs(true_fvc - pred_fvc)

        # Clip error at 1000ml (MAX_FVC_ERROR)
        delta_clipped = torch.clamp(delta, max=self.max_fvc_error)

        # Clip confidence at 70ml (MIN_CONFIDENCE)
        # Note: pred_sigma is assumed to be positive (e.g., via Softplus in model)
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_confidence)

        # Calculate metric terms
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=pred_fvc.device))

        term1 = (sqrt_2 * delta_clipped) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Loss is the sum of terms (negative of the metric)
        loss = term1 + term2

        return torch.mean(loss)
