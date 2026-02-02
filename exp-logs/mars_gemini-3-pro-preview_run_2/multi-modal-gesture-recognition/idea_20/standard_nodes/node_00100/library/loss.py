import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss for temporal smoothing.
    Penalizes variations in probability distributions between adjacent frames,
    but truncates the penalty for large changes (likely true boundaries) to allow sharp transitions.
    """

    def __init__(self, threshold=0.15):
        super(TMSELoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, probs, mask):
        """
        Args:
            probs (torch.Tensor): (B, T, C) Softmax probabilities.
            mask (torch.Tensor): (B, T) Sequence mask.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Calculate temporal difference: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared difference
        diff_sq = diff.pow(2)

        # Truncate the loss (allow sharp jumps)
        # If diff^2 > threshold^2, the gradient becomes 0, allowing the jump.
        loss_tensor = torch.clamp(diff_sq, min=0, max=self.threshold_sq)

        # Create mask for transitions
        # A transition is valid if both t and t-1 are valid
        # Shape: (B, T-1)
        mask_trans = mask[:, 1:] * mask[:, :-1]

        # Expand mask for channels: (B, T-1, C)
        mask_trans_expanded = mask_trans.unsqueeze(-1)

        # Apply mask
        masked_loss = loss_tensor * mask_trans_expanded

        # Normalize
        # Avoid division by zero
        num_valid = mask_trans_expanded.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=probs.device)


class DSG_Loss(nn.Module):
    """
    Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network Loss.
    Aggregates losses from all three stages (Deep Supervision).

    Components per stage:
    1. Weighted Classification Loss (NLL on Log-Probs)
    2. Boundary Binary Cross Entropy
    3. Truncated MSE for Smoothing
    """

    def __init__(self):
        super(DSG_Loss, self).__init__()

        # Load weights from config and move to device dynamically in forward
        self.class_weights = Config.CLASS_WEIGHTS

        # Loss components
        self.tmse = TMSELoss(threshold=Config.TMSE_THRESHOLD)

        # Weights for different loss terms
        self.w_bnd = Config.BOUNDARY_LOSS_WEIGHT
        self.w_tmse = Config.TMSE_WEIGHT

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs (dict): Dictionary containing outputs from all stages.
                Keys: 'stageX_cls' (B, T, C), 'stageX_bnd' (B, T, 1)
            targets (dict): Dictionary containing ground truth.
                Keys: 'labels' (B, T), 'boundaries' (B, T), 'mask' (B, T)

        Returns:
            torch.Tensor: Total aggregated loss.
            dict: Dictionary of individual loss components for logging.
        """
        labels = targets["labels"]  # (B, T) Long
        boundaries = targets["boundaries"]  # (B, T) Float
        mask = targets["mask"]  # (B, T) Float

        # Ensure class weights are on the correct device
        if self.class_weights.device != labels.device:
            self.class_weights = self.class_weights.to(labels.device)

        total_loss = 0.0
        metrics = {}

        # Iterate over all 3 stages
        for stage in range(1, Config.NUM_STAGES + 1):
            prefix = f"stage{stage}"

            # --- 1. Classification Loss ---
            # Model outputs probabilities (Softmax applied), so we use NLLLoss with log
            pred_cls = model_outputs[f"{prefix}_cls"]  # (B, T, C)

            # Numerical stability for log
            log_probs = torch.log(pred_cls + 1e-7)

            # NLLLoss expects (B, C, T) or (N, C)
            # We reshape to (N, C) for simplicity
            B, T, C = log_probs.shape
            log_probs_flat = log_probs.reshape(-1, C)
            labels_flat = labels.reshape(-1)

            # Calculate unreduced loss
            cls_loss_flat = F.nll_loss(
                log_probs_flat, labels_flat, weight=self.class_weights, reduction="none"
            )

            # Reshape back to (B, T) and apply mask
            cls_loss_seq = cls_loss_flat.reshape(B, T)
            masked_cls_loss = (cls_loss_seq * mask).sum() / (mask.sum() + 1e-7)

            # --- 2. Boundary Loss ---
            pred_bnd = model_outputs[f"{prefix}_bnd"].squeeze(-1)  # (B, T)

            # BCE Loss
            bnd_loss_seq = F.binary_cross_entropy(
                pred_bnd, boundaries, reduction="none"
            )
            masked_bnd_loss = (bnd_loss_seq * mask).sum() / (mask.sum() + 1e-7)

            # --- 3. Smoothing Loss (TMSE) ---
            # Applied to class probabilities
            tmse_loss_val = self.tmse(pred_cls, mask)

            # --- Aggregate Stage Loss ---
            stage_loss = (
                masked_cls_loss
                + (self.w_bnd * masked_bnd_loss)
                + (self.w_tmse * tmse_loss_val)
            )

            total_loss += stage_loss

            # Log metrics
            metrics[f"{prefix}_loss"] = stage_loss.item()
            metrics[f"{prefix}_cls"] = masked_cls_loss.item()
            metrics[f"{prefix}_bnd"] = masked_bnd_loss.item()
            metrics[f"{prefix}_tmse"] = tmse_loss_val.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
