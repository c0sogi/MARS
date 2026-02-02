import numpy as np
import torch
from library.config import set_seed

# Alias seed_everything to the provided set_seed function for consistency and code reuse
seed_everything = set_seed


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def metric_function(true_fvc, pred_fvc, confidence):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.ndarray or torch.Tensor): Ground truth FVC values.
        pred_fvc (np.ndarray or torch.Tensor): Predicted FVC values.
        confidence (np.ndarray or torch.Tensor): Predicted confidence (sigma) values.

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.detach().cpu().numpy()

    # Flatten arrays to 1D to ensure element-wise operations work correctly
    true_fvc = true_fvc.reshape(-1).astype(float)
    pred_fvc = pred_fvc.reshape(-1).astype(float)
    confidence = confidence.reshape(-1).astype(float)

    # Apply clipping constraints defined in the metric
    # Confidence is clipped at 70 ml
    sigma_clipped = np.maximum(confidence, 70)

    # Error (Delta) is clipped at 1000 ml
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, 1000)

    # Calculate the metric
    # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score over the batch
    return np.mean(metric)
