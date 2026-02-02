import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
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
    Implements the modified Laplace Log Likelihood metric as a loss function.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, the loss is defined as -metric:
        loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.sigma_clip = Config.SIGMA_CLIP
        self.max_delta = Config.MAX_DELTA
        # Register sqrt(2) as a buffer so it moves to the correct device automatically
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, outputs, targets):
        """
        Computes the loss.

        Args:
            outputs (torch.Tensor): Predictions of shape (Batch_Size, 2).
                                    Column 0: Predicted FVC
                                    Column 1: Predicted Confidence (Sigma)
            targets (torch.Tensor): Ground truth FVC of shape (Batch_Size,) or (Batch_Size, 1).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Ensure targets are (B, 1)
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        # Extract predictions
        fvc_pred = outputs[:, 0].view(-1, 1)
        sigma_pred = outputs[:, 1].view(-1, 1)

        # Apply constraints defined in the metric
        # 1. Clip confidence (sigma) at 70ml
        sigma_clipped = torch.clamp(sigma_pred, min=self.sigma_clip)

        # 2. Calculate absolute error (delta)
        delta = torch.abs(targets - fvc_pred)

        # 3. Clip error at 1000ml
        delta_clipped = torch.clamp(delta, max=self.max_delta)

        # Calculate loss components
        # Loss = (sqrt(2) * delta_clipped) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Sum terms to get negative metric (Loss)
        loss = term1 + term2

        # Return mean loss over the batch
        return torch.mean(loss)
