import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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
    Custom loss function based on the modified Laplace Log Likelihood metric.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Since optimizers minimize loss, this class calculates the negative of the metric:
        Loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.min_confidence = Config.MIN_CONFIDENCE
        self.max_error = Config.MAX_ERROR
        # Register sqrt(2) as a buffer so it moves with the model to GPU
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, targets):
        """
        Calculates the loss.

        Args:
            preds (torch.Tensor): Tensor of shape (batch_size, 2).
                                  preds[:, 0] is the predicted FVC.
                                  preds[:, 1] is the predicted Confidence (Sigma).
            targets (torch.Tensor): Tensor of shape (batch_size,) or (batch_size, 1)
                                    containing the ground truth FVC.

        Returns:
            torch.Tensor: The mean loss over the batch.
        """
        # Separate predictions
        pred_fvc = preds[:, 0]
        pred_sigma = preds[:, 1]

        # Flatten targets to match pred_fvc shape
        true_fvc = targets.view(-1)

        # Calculate absolute error (Delta)
        delta = torch.abs(true_fvc - pred_fvc)

        # Clip error at 1000 ml
        delta_clipped = torch.clamp(delta, max=self.max_error)

        # Clip confidence at 70 ml
        # pred_sigma is assumed to be positive (e.g., via Softplus in the model)
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_confidence)

        # Calculate Loss components
        # Term 1: (sqrt(2) * Delta) / Sigma
        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped

        # Term 2: ln(sqrt(2) * Sigma)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # Sum components to get Loss (Negative Metric)
        loss = term1 + term2

        return torch.mean(loss)


def calculate_metric(preds, targets):
    """
    Calculates the raw competition metric (higher is better).
    Useful for logging validation scores.

    Args:
        preds (torch.Tensor): Predicted FVC and Sigma (B, 2).
        targets (torch.Tensor): True FVC (B,).

    Returns:
        float: The mean metric score.
    """
    # We can reuse the loss class logic
    # Metric = -Loss
    with torch.no_grad():
        # Instantiate loss on the correct device
        loss_fn = LaplaceLogLikelihoodLoss()
        loss_fn.to(preds.device)

        loss = loss_fn(preds, targets)

        # Return negative loss (the actual metric value)
        return -loss.item()
