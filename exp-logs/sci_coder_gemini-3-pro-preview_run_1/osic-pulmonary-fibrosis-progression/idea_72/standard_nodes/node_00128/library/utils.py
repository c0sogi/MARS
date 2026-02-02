import numpy as np
import torch
import torch.nn as nn
from library.config import Config, seed_everything


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Custom Loss function based on the competition metric:
    Modified Laplace Log Likelihood.

    The metric is defined as:
    sigma_clipped = max(sigma, 70)
    delta = min(|true - pred|, 1000)
    metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, we minimize the negative metric (Loss).
    Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.max_error = Config.MAX_ERROR
        self.min_confidence = Config.MIN_CONFIDENCE

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the loss for training.

        Args:
            pred_fvc (Tensor): Predicted FVC values.
            pred_sigma (Tensor): Predicted Confidence (sigma) values.
            true_fvc (Tensor): Ground truth FVC values.

        Returns:
            Tensor: Scalar loss value (mean over batch).
        """
        # Ensure shapes match for broadcasting
        if pred_fvc.shape != true_fvc.shape:
            true_fvc = true_fvc.view_as(pred_fvc)
        if pred_sigma.shape != pred_fvc.shape:
            pred_sigma = pred_sigma.view_as(pred_fvc)

        # Clip sigma (Confidence) to reflect approximate measurement uncertainty
        # We assume the model outputs positive sigma (e.g. via Softplus)
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_confidence)

        # Calculate absolute error and clip it to avoid large errors penalizing results
        abs_error = torch.abs(true_fvc - pred_fvc)
        delta = torch.clamp(abs_error, max=self.max_error)

        # Calculate Loss components
        sqrt_2 = np.sqrt(2)
        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


def score(pred_fvc, pred_sigma, true_fvc):
    """
    Calculates the competition metric for evaluation.
    Metric values will be negative and higher is better.

    Args:
        pred_fvc: Predicted FVC (Tensor or numpy array).
        pred_sigma: Predicted Confidence (Tensor or numpy array).
        true_fvc: Ground truth FVC (Tensor or numpy array).

    Returns:
        float: The average metric score.
    """
    # Convert Tensors to Numpy arrays if necessary
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(pred_sigma, torch.Tensor):
        pred_sigma = pred_sigma.detach().cpu().numpy()
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()

    # Flatten arrays to ensure 1D processing
    pred_fvc = pred_fvc.flatten()
    pred_sigma = pred_sigma.flatten()
    true_fvc = true_fvc.flatten()

    # Constants
    max_error = Config.MAX_ERROR
    min_confidence = Config.MIN_CONFIDENCE

    # Metric calculation logic
    sigma_clipped = np.maximum(pred_sigma, min_confidence)
    abs_error = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(abs_error, max_error)

    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
