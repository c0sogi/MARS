import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# Set fixed seeds for reproducibility
torch.manual_seed(Config.SEED)


class CombinedLoss(nn.Module):
    """
    Computes the combined objective function for the Cascaded Refinement Network.

    Components:
    1. Weighted Cross-Entropy Loss (Stage 1): Supervision for the Bi-GRU Encoder.
    2. Weighted Cross-Entropy Loss (Stage 2): Supervision for the Holo-Refinement TCN.
    3. Truncated Log-Space Smoothing Loss (Stage 2): Penalizes high-frequency jitter
       in predictions while allowing valid transitions.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # Load class weights from config (Background=0.2, Others=1.0)
        # We register this as a buffer so it moves to the correct device with the model
        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        self.lambda_smooth = Config.SMOOTHING_LAMBDA

        # Truncation threshold for smoothing loss.
        # A squared log-diff of 1.0 allows for reasonable probability shifts (transitions)
        # without incurring massive penalties, while small jitter is penalized fully.
        self.truncation_threshold = 1.0

    def forward(self, stage1_logits, stage2_logits, targets):
        """
        Args:
            stage1_logits: (Batch, Time, Num_Classes) - Output from BiGRU
            stage2_logits: (Batch, Time, Num_Classes) - Output from TCN
            targets: (Batch, Time) - Ground truth labels

        Returns:
            dict: Contains 'loss' (total scalar) and individual components.
        """
        # 1. Prepare inputs for CrossEntropyLoss
        # CE expects (Batch, Classes, Time) for multi-dimensional loss
        s1_input = stage1_logits.permute(0, 2, 1)
        s2_input = stage2_logits.permute(0, 2, 1)

        # 2. Compute Classification Losses
        loss_ce1 = self.ce_loss(s1_input, targets)
        loss_ce2 = self.ce_loss(s2_input, targets)

        # 3. Compute Truncated Log-Space Smoothing Loss (Stage 2 only)
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(stage2_logits, dim=2)  # (Batch, Time, Classes)

        # Calculate difference between frame t and t-1
        # Shape: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        sq_diff = diff**2

        # Truncate the error to avoid penalizing valid class transitions too heavily
        truncated_sq_diff = torch.clamp(sq_diff, max=self.truncation_threshold)

        # Mean over batch, time, and classes
        loss_smooth = torch.mean(truncated_sq_diff)

        # 4. Aggregate
        total_loss = loss_ce1 + loss_ce2 + (self.lambda_smooth * loss_smooth)

        return {
            "loss": total_loss,
            "ce1": loss_ce1,
            "ce2": loss_ce2,
            "smooth": loss_smooth,
        }
