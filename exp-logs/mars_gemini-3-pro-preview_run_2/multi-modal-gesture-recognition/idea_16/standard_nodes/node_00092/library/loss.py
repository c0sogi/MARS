import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss for probability smoothing.
    Penalizes rapid changes in frame-wise probability distributions to enforce
    temporal consistency.
    """

    def __init__(self, threshold=1.0):
        """
        Args:
            threshold (float): The maximum allowed difference before truncation.
                               Defaults to 1.0, which for probabilities [0,1]
                               effectively disables truncation (standard MSE),
                               adhering to the 'do not clamp' instruction.
        """
        super(TMSELoss, self).__init__()
        self.threshold = threshold**2

    def forward(self, probs, mask):
        """
        Args:
            probs: (Batch, Time, Classes) - Softmax probabilities
            mask: (Batch, Time) - Valid frames mask
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate temporal differences: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        mse = diff**2

        # Truncation (Clamp the error)
        # With threshold=1.0, this is effectively standard MSE for probabilities.
        tmse = torch.clamp(mse, min=0, max=self.threshold)

        # Masking
        # A transition is valid if both t and t-1 are valid.
        mask_t = mask[:, 1:]  # Mask for t
        mask_t_prev = mask[:, :-1]  # Mask for t-1
        valid_transitions = mask_t & mask_t_prev  # (B, T-1)

        # Expand mask for classes: (B, T-1, 1)
        valid_transitions = valid_transitions.unsqueeze(-1).float()

        # Compute mean loss over valid transitions
        # Denominator adds epsilon to prevent division by zero for empty sequences
        loss = (tmse * valid_transitions).sum() / (
            valid_transitions.sum() * probs.shape[2] + 1e-7
        )

        return loss


class CombinedLoss(nn.Module):
    """
    Aggregates Classification and Smoothing losses across all model stages
    (Deep Supervision).
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        # Load class weights and register as buffer for device management
        self.register_buffer("class_weights", Config.get_class_weights())

        # Initialize Smoothing Loss
        self.tmse = TMSELoss(threshold=1.0)
        self.tmse_weight = Config.TMSE_WEIGHT

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: Dict with keys 'stage1_cls', 'stage2_cls', 'stage3_cls'
                         containing Softmax probabilities (B, T, C).
            targets: (Batch, Time) - Class indices (Long).
            mask: (Batch, Time) - Valid frames (Bool).
        Returns:
            torch.Tensor: Total aggregated loss.
        """
        total_loss = 0.0

        # Iterate over all supervised stages
        stages = ["stage1_cls", "stage2_cls", "stage3_cls"]

        for stage_key in stages:
            if stage_key not in predictions:
                continue

            probs = predictions[stage_key]  # (B, T, C)

            # -----------------------------------------------------------------
            # 1. Classification Loss (Weighted NLL)
            # -----------------------------------------------------------------
            # Inputs are Softmax probabilities, so we take Log to use NLLLoss.
            # Add epsilon for numerical stability.
            log_probs = torch.log(probs + 1e-7)

            # NLLLoss expects (B, C, T)
            log_probs_perm = log_probs.permute(0, 2, 1)

            # Compute element-wise loss with ignore_index for padding (-1)
            # targets: (B, T)
            ce_loss = F.nll_loss(
                log_probs_perm,
                targets,
                weight=self.class_weights,
                reduction="none",
                ignore_index=-1,
            )  # Output: (B, T)

            # Apply Mask manually (redundant if ignore_index works, but safer)
            masked_ce = (ce_loss * mask.float()).sum() / (mask.float().sum() + 1e-7)

            # -----------------------------------------------------------------
            # 2. Smoothing Loss (TMSE)
            # -----------------------------------------------------------------
            tmse_loss = self.tmse(probs, mask)

            # -----------------------------------------------------------------
            # 3. Aggregate
            # -----------------------------------------------------------------
            total_loss += masked_ce + self.tmse_weight * tmse_loss

        return total_loss
