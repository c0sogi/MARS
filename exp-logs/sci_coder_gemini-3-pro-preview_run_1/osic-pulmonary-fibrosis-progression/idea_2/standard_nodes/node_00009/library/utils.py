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
    Custom Loss function based on the competition metric:
    Modified Laplace Log Likelihood.

    The model predicts parameters defining the trajectory and uncertainty:
    - alpha: Slope of FVC decline
    - sigma_base: Uncertainty at t=0
    - sigma_growth: Rate of uncertainty increase over time

    Loss = -Metric (since we want to maximize the metric).
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.q_clip = Config.Q_CLIP
        self.max_err = Config.MAX_ERR
        # Pre-compute constant
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, preds, targets, initial_fvc, time_delta):
        """
        Args:
            preds (Tensor): Shape (B, 3) containing [alpha, sigma_base, sigma_growth]
            targets (Tensor): Shape (B,) containing true FVC
            initial_fvc (Tensor): Shape (B,) containing baseline FVC
            time_delta (Tensor): Shape (B,) containing weeks relative to baseline

        Returns:
            loss (Tensor): Scalar loss value
        """
        # Extract predicted parameters
        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # 1. Calculate Predicted FVC
        # Linear model: FVC(t) = Baseline + alpha * t
        fvc_pred = initial_fvc + alpha * time_delta

        # 2. Calculate Predicted Confidence (Sigma)
        # Confidence(t) = sigma_base + sigma_growth * |t|
        # We use abs() to ensure positive contributions to uncertainty
        confidence = torch.abs(sigma_base) + torch.abs(sigma_growth) * torch.abs(
            time_delta
        )

        # 3. Apply Metric Rules
        # Clip confidence at 70ml
        sigma_clipped = torch.clamp(confidence, min=self.q_clip)

        # Calculate absolute error and clip at 1000ml
        abs_err = torch.abs(targets - fvc_pred)
        delta = torch.clamp(abs_err, max=self.max_err)

        # 4. Compute Metric
        # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        sqrt_2 = self.sqrt_2.to(preds.device)

        metric_term1 = (sqrt_2 * delta) / sigma_clipped
        metric_term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = metric_term1 + metric_term2

        return torch.mean(loss)


def calculate_metric_numpy(preds, targets, initial_fvc, time_delta):
    """
    Calculates the competition metric using NumPy arrays.
    Useful for validation and evaluation.

    Args:
        preds (np.array): Shape (N, 3) [alpha, sigma_base, sigma_growth]
        targets (np.array): Shape (N,) True FVC
        initial_fvc (np.array): Shape (N,) Baseline FVC
        time_delta (np.array): Shape (N,) Weeks relative to baseline

    Returns:
        score (float): The mean Laplace Log Likelihood score (higher is better)
    """
    alpha = preds[:, 0]
    sigma_base = preds[:, 1]
    sigma_growth = preds[:, 2]

    # Predict FVC and Confidence
    fvc_pred = initial_fvc + alpha * time_delta
    confidence = np.abs(sigma_base) + np.abs(sigma_growth) * np.abs(time_delta)

    # Clip values
    sigma_clipped = np.maximum(confidence, Config.Q_CLIP)

    abs_err = np.abs(targets - fvc_pred)
    delta = np.minimum(abs_err, Config.MAX_ERR)

    # Calculate Metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
