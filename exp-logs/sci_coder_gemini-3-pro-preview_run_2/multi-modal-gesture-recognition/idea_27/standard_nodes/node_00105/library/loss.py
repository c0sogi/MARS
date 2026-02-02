import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss for temporal smoothing.

    As per the specific task requirements ("Unclamped Probability-Space Smoothing"),
    this implementation calculates the Mean Squared Error on the temporal differences
    of the Softmax probability distributions without applying a truncation threshold.
    It strictly enforces masking to prevent smoothing across padding boundaries.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Probability distributions of shape (Batch, Time, Classes).
            mask (torch.Tensor): Boolean mask of shape (Batch, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate squared difference between adjacent frames: (P_t - P_{t-1})^2
        # Shape: (Batch, Time-1, Classes)
        diff = (x[:, 1:, :] - x[:, :-1, :]) ** 2

        # Sum over classes to get total squared difference per time step
        # Shape: (Batch, Time-1)
        diff_sum = diff.sum(dim=2)

        # Create mask for valid transitions (both current and previous frame must be valid)
        # Shape: (Batch, Time-1)
        mask_valid = mask[:, 1:] & mask[:, :-1]

        # Avoid division by zero if sequence is too short or fully masked
        if mask_valid.sum() == 0:
            return torch.tensor(0.0, device=x.device)

        # Average over valid transitions only
        loss = diff_sum[mask_valid].mean()

        return loss


class HierarchicalLoss(nn.Module):
    """
    Multi-component objective function for the GHG-CRCN model.

    Computes a weighted sum of:
    1. Weighted Cross-Entropy (Classification) - Handles class imbalance (0.1 vs 1.0).
    2. Binary Cross-Entropy (Boundary Detection) - Uses sharp targets.
    3. Binary Cross-Entropy (Foreground Presence) - Reinforces background suppression.
    4. TMSE (Temporal Smoothing) - Enforces smoothness in probability space.

    The loss is applied to every stage of the model (Deep Supervision).
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()

        # Classification Loss:
        # We use NLLLoss because we will manually apply log to the softmax probabilities.
        # This allows us to work with the pre-computed probabilities from the model output.
        self.nll_loss = nn.NLLLoss(weight=Config.CLASS_WEIGHTS, reduction="mean")

        # Binary Classification Losses for Boundary and Foreground
        self.bce_loss = nn.BCELoss(reduction="mean")

        # Smoothing Loss
        self.tmse_loss = TMSELoss()

    def forward(self, stage_outputs, targets_cls, targets_bnd, targets_fg, mask):
        """
        Args:
            stage_outputs (list[torch.Tensor]): List of model outputs from each stage.
                                                Each tensor has shape (Batch, Time, NumClasses + 2).
            targets_cls (torch.Tensor): Ground truth class labels (Batch, Time).
            targets_bnd (torch.Tensor): Ground truth boundary labels (Batch, Time).
            targets_fg (torch.Tensor): Ground truth foreground labels (Batch, Time).
            mask (torch.Tensor): Boolean sequence mask (Batch, Time).

        Returns:
            torch.Tensor: Total aggregated loss across all stages.
        """
        total_loss = 0.0
        device = targets_cls.device

        # Ensure class weights are on the correct device (e.g., if moved to GPU)
        if self.nll_loss.weight is not None and self.nll_loss.weight.device != device:
            self.nll_loss.weight = self.nll_loss.weight.to(device)

        # Flatten mask for selecting valid elements from tensors
        # Shape: (Batch * Time)
        flat_mask = mask.view(-1)

        # Pre-process targets: Flatten and select only valid elements
        t_cls_flat = targets_cls.view(-1)[flat_mask]
        t_bnd_flat = targets_bnd.view(-1)[flat_mask]
        t_fg_flat = targets_fg.view(-1)[flat_mask]

        # Safety check: if batch is empty or fully masked
        if t_cls_flat.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Iterate over outputs from each stage (Deep Supervision)
        for out in stage_outputs:
            # out shape: (Batch, Time, NumClasses + 1 + 1)

            # --- Split Model Outputs ---
            # 1. Class Probabilities
            probs_cls = out[:, :, : Config.NUM_CLASSES]

            # 2. Boundary Probabilities
            probs_bnd = out[:, :, Config.NUM_CLASSES : Config.NUM_CLASSES + 1]

            # 3. Foreground Probabilities
            probs_fg = out[:, :, Config.NUM_CLASSES + 1 :]

            # --- 1. Classification Loss ---
            # Flatten and select valid frames
            p_cls_flat = probs_cls.reshape(-1, Config.NUM_CLASSES)[flat_mask]

            # Apply log for NLLLoss (add epsilon for numerical stability)
            log_probs_cls = torch.log(p_cls_flat + 1e-8)
            loss_cls = self.nll_loss(log_probs_cls, t_cls_flat)

            # --- 2. Boundary Loss ---
            p_bnd_flat = probs_bnd.view(-1)[flat_mask]
            loss_bnd = self.bce_loss(p_bnd_flat, t_bnd_flat)

            # --- 3. Foreground Loss ---
            p_fg_flat = probs_fg.view(-1)[flat_mask]
            loss_fg = self.bce_loss(p_fg_flat, t_fg_flat)

            # --- 4. Smoothing Loss ---
            # TMSE handles masking internally to preserve temporal adjacency logic
            loss_smooth = self.tmse_loss(probs_cls, mask)

            # --- Aggregation ---
            # Weighted sum of components
            stage_loss = (
                Config.W_CLS * loss_cls
                + Config.W_BND * loss_bnd
                + Config.W_FG * loss_fg
                + Config.W_SMOOTH * loss_smooth
            )

            total_loss += stage_loss

        return total_loss
