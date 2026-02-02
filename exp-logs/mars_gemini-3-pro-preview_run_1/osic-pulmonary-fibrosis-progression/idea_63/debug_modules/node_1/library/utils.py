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
    if torch.cuda.is_available():
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


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss.

    Metric Definition:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Loss Definition (to be minimized):
        Loss = -metric
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_abs_error = Config.max_absolute_error
        self.min_confidence = Config.min_confidence

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Shape (Batch, 2).
                                  Column 0 is FVC prediction.
                                  Column 1 is Confidence (Sigma) prediction.
            targets (torch.Tensor): Shape (Batch,) or (Batch, 1).
                                    True FVC values.
        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Separate predictions
        fvc_pred = preds[:, 0]
        sigma_pred = preds[:, 1]

        # Ensure targets are 1D matching the batch dimension of columns
        fvc_true = targets.view(-1)

        # Calculate Delta: |True - Pred|
        delta = torch.abs(fvc_true - fvc_pred)
        # Clip Delta at 1000 ml
        delta = torch.clamp(delta, max=self.max_abs_error)

        # Clip Sigma at 70 ml
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_confidence)

        # Calculate Loss terms
        # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=preds.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)
