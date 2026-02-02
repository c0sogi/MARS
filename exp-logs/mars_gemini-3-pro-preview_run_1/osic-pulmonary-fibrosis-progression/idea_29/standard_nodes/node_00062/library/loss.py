import torch
import torch.nn as nn
from library.config import Config


class RobustLaplaceLoss(nn.Module):
    """
    Implements the robust negative modified Laplace Log Likelihood loss.

    This loss function reconstructs the FVC predictions from the trajectory parameters
    output by CVRNet and computes the metric-compliant loss.

    Metric Definition:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Optimization Objective:
        Loss = -Metric
    """

    def __init__(self):
        super(RobustLaplaceLoss, self).__init__()
        # Load constraints from Config
        self.max_error = Config.MAX_ERROR  # 1000 ml
        self.min_confidence = Config.MIN_CONFIDENCE  # 70 ml

        # Register sqrt(2) as a buffer so it moves to GPU automatically with the module
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, preds, targets, meta):
        """
        Calculates the loss based on model predictions and ground truth.

        Args:
            preds (tuple): A tuple containing (alpha, sigma_base, sigma_growth)
                           output by the CVRNet model. Each is a tensor of shape (B,).
            targets (torch.Tensor): Ground truth FVC values of shape (B,).
            meta (torch.Tensor): Metadata tensor of shape (B, 2) containing
                                 [Baseline_FVC, Week_Diff] for reconstruction.

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # Unpack model predictions
        alpha, sigma_base, sigma_growth = preds

        # Unpack metadata for reconstruction
        # meta[:, 0] -> Baseline_FVC
        # meta[:, 1] -> Week_Diff (Week - Baseline_Week)
        base_fvc = meta[:, 0]
        week_diff = meta[:, 1]

        # 1. Reconstruct FVC Prediction (Trajectory Logic)
        # FVC_pred = Baseline + Slope * (Week - Baseline_Week)
        fvc_pred = base_fvc + alpha * week_diff

        # 2. Reconstruct Confidence/Sigma (Uncertainty Logic)
        # Sigma = Sigma_base + Sigma_growth * |Week - Baseline_Week|
        # Note: sigma_base and sigma_growth are already positive (Softplus in model)
        sigma = sigma_base + sigma_growth * torch.abs(week_diff)

        # 3. Apply Metric Constraints (Differentiable)

        # Clip Confidence: sigma_clipped = max(sigma, 70)
        sigma_clipped = torch.clamp(sigma, min=self.min_confidence)

        # Calculate Absolute Error
        abs_error = torch.abs(targets - fvc_pred)

        # Clip Error: delta = min(|True - Pred|, 1000)
        # Using clamp(max=1000) ensures gradients are 0 for errors > 1000,
        # providing the required robustness against outliers.
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute Negative Log Likelihood
        # The metric is: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # We minimize the negative: (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)
