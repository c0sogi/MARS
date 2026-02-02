import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TruncatedMSE(nn.Module):
    """
    Truncated Mean Squared Error (MSE) loss applied to log-probabilities.
    Penalizes rapid changes in frame-wise predictions to enforce temporal smoothness.

    Formula: mean( clamp( (log_p[t] - log_p[t-1])^2, max=tau^2 ) )
    """

    def __init__(self, threshold=4.0):
        super(TruncatedMSE, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, log_probs):
        """
        Args:
            log_probs: (Batch, Classes, Time)
        Returns:
            loss: Scalar tensor
        """
        # Calculate difference between adjacent frames: log_p[:, :, t] - log_p[:, :, t-1]
        # Slice input: [:, :, 1:] and [:, :, :-1]
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Square the differences
        diff_sq = diff**2

        # Clamp the squared differences to threshold^2
        # This prevents outliers (sudden jumps) from dominating the gradient
        clamp_val = self.threshold**2
        diff_clamped = torch.clamp(diff_sq, max=clamp_val)

        # Average over all dimensions
        loss = torch.mean(diff_clamped)

        return loss


class CascadedLoss(nn.Module):
    """
    Cascaded Loss function for Deep Supervision.
    Aggregates Weighted Cross-Entropy (NLL) and Truncated MSE Smoothing
    across all model stages.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Load class weights from Config and move to GPU if available
        # Weights: 0.2 for background, 1.0 for others
        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
        if torch.cuda.is_available():
            weights = weights.cuda()

        # Use NLLLoss because we will compute log_probs manually from model output probabilities
        self.nll_loss = nn.NLLLoss(weight=weights, reduction="mean")

        self.smooth_loss = TruncatedMSE()
        self.lambda_smooth = Config.LAMBDA_SMOOTH

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing outputs from each stage.
                     {'stage1': (B, T, C), 'stage2': (B, T, C), ...}
            targets: Ground truth labels (B, T)

        Returns:
            total_loss: Scalar tensor for backprop
            metrics: Dictionary of loss components for logging
        """
        total_loss = 0.0
        metrics = {}

        # Iterate over all stages present in the output
        for stage_name, probs in outputs.items():
            # probs shape: (Batch, Time, Classes)
            # targets shape: (Batch, Time)

            # 1. Prepare Data
            # Permute probs to (Batch, Classes, Time) for NLLLoss and Smoothing
            probs_permuted = probs.transpose(1, 2)

            # Compute Log Probabilities (add epsilon for numerical stability)
            log_probs = torch.log(torch.clamp(probs_permuted, min=1e-7))

            # 2. Classification Loss (Weighted NLL)
            cls_loss = self.nll_loss(log_probs, targets)

            # 3. Smoothing Loss (Truncated MSE)
            smooth_l = self.smooth_loss(log_probs)

            # 4. Stage Total
            stage_loss = cls_loss + (self.lambda_smooth * smooth_l)

            # Accumulate
            total_loss += stage_loss

            # Record metrics (detach for logging to avoid graph retention)
            metrics[f"{stage_name}_loss"] = stage_loss.item()
            metrics[f"{stage_name}_cls"] = cls_loss.item()
            metrics[f"{stage_name}_smooth"] = smooth_l.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
