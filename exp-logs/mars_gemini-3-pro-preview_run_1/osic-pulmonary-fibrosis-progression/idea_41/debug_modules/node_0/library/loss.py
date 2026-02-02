import torch
import torch.nn as nn
from library.config import Config


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the modified Laplace Log Likelihood loss function for the
    OSIC Pulmonary Fibrosis Progression task.

    The competition metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Since we want to maximize the metric, we minimize the negative metric (Loss):
        Loss = (sqrt(2) * delta / sigma_clipped) + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        # Load constraints from configuration
        self.max_error = float(Config.MAX_ERROR_CLIP)
        self.min_sigma = float(Config.SIGMA_CLIP)

        # Register sqrt(2) as a buffer so it automatically moves to the correct device (CPU/GPU)
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        """
        Calculates the loss for a batch of predictions.

        Args:
            pred_fvc (torch.Tensor): Predicted FVC values (Batch_Size,).
            pred_sigma (torch.Tensor): Predicted Confidence/Sigma values (Batch_Size,).
            true_fvc (torch.Tensor): Ground truth FVC values (Batch_Size,).

        Returns:
            torch.Tensor: The mean loss over the batch (scalar).
        """
        # Ensure inputs are float for calculation
        pred_fvc = pred_fvc.float()
        pred_sigma = pred_sigma.float()
        true_fvc = true_fvc.float()

        # 1. Calculate Absolute Error (Delta)
        delta = torch.abs(true_fvc - pred_fvc)

        # 2. Apply Metric Constraints
        # Clip error at 1000 ml: min(|True - Pred|, 1000)
        delta_clipped = torch.clamp(delta, max=self.max_error)

        # Clip confidence at 70 ml: max(Sigma, 70)
        sigma_clipped = torch.clamp(pred_sigma, min=self.min_sigma)

        # 3. Calculate Loss Terms
        # Term 1: (sqrt(2) * Delta_clipped) / Sigma_clipped
        term1 = (self.sqrt_2 * delta_clipped) / sigma_clipped

        # Term 2: ln(sqrt(2) * Sigma_clipped)
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        # 4. Combine Terms
        # Loss = Term1 + Term2 (which is equivalent to -Metric)
        loss = term1 + term2

        # Return mean loss over the batch
        return torch.mean(loss)
