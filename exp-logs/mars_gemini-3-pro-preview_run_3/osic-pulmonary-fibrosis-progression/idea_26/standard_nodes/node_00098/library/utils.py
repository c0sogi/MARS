import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Custom loss function maximizing the Laplace Log Likelihood.
    Loss = (sqrt(2) * |true - pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self, reduction="mean"):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.reduction = reduction

    def forward(self, preds, target):
        """
        Args:
            preds: Tensor of shape (Batch, 2).
                   Column 0: Predicted FVC
                   Column 1: Predicted Confidence (Sigma)
            target: Tensor of shape (Batch, 1) or (Batch,). True FVC.
        """
        # Ensure target is the correct shape
        if target.dim() == 1:
            target = target.view(-1, 1)

        fvc_pred = preds[:, 0].view(-1, 1)
        sigma_pred = preds[:, 1].view(-1, 1)

        # Calculate absolute error
        delta = torch.abs(target - fvc_pred)

        # Calculate loss terms
        # Note: sigma_pred is assumed to be positive (handled by model architecture)
        term1 = (torch.sqrt(torch.tensor(2.0).to(preds.device)) * delta) / sigma_pred
        term2 = torch.log(torch.sqrt(torch.tensor(2.0).to(preds.device)) * sigma_pred)

        loss = term1 + term2

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class MetricMonitor:
    """
    Tracks the average loss and the competition metric.

    Competition Metric:
    sigma_clipped = max(sigma, 70)
    delta = min(|true - pred|, 1000)
    metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val_loss_sum = 0
        self.metric_sum = 0
        self.count = 0

    def update(self, loss_val, preds, target):
        """
        Args:
            loss_val: float, the loss value for the current batch.
            preds: Tensor of shape (Batch, 2) [FVC, Sigma].
            target: Tensor of shape (Batch, 1) or (Batch,).
        """
        batch_size = preds.shape[0]

        # Update Loss
        self.val_loss_sum += loss_val * batch_size

        # Calculate Metric
        # Move to CPU/Numpy for metric calculation to handle clipping easily
        preds_np = preds.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy().flatten()

        # Inverse Transform to Real Units (ml) for Metric Calculation
        # Cite solution_lesson_node_00066: Exact Loss Alignment / Metric Validity
        fvc_mean = Config.STATS["fvc_mean"]
        fvc_std = Config.STATS["fvc_std"]

        # Preds are [Mu_scaled, Sigma_scaled]
        # Target is [FVC_scaled]

        real_mu = preds_np[:, 0] * fvc_std + fvc_mean
        real_sigma = preds_np[:, 1] * fvc_std
        real_target = target_np * fvc_std + fvc_mean

        # Apply competition metric constraints
        # Clip sigma at 70 ml
        sigma_clipped = np.maximum(real_sigma, Config.MIN_UNCERTAINTY)

        # Calculate Delta and clip at 1000 ml
        delta = np.abs(real_target - real_mu)
        delta_clipped = np.minimum(delta, Config.MAX_ERROR)

        # Metric Formula
        metric = -(np.sqrt(2) * delta_clipped) / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )

        self.metric_sum += np.sum(metric)
        self.count += batch_size

    def get_avg_loss(self):
        if self.count == 0:
            return 0.0
        return self.val_loss_sum / self.count

    def get_avg_score(self):
        if self.count == 0:
            return 0.0
        return self.metric_sum / self.count
