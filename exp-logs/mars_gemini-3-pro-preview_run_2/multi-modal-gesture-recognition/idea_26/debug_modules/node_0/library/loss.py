import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss for temporal smoothing.
    Penalizes rapid changes in probabilities between consecutive frames.
    """

    def __init__(self, threshold=4.0):
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, probabilities, mask):
        """
        Args:
            probabilities: (B, T, C) Softmax probabilities.
            mask: (B, T) Sequence mask.
        Returns:
            Scalar loss.
        """
        # Calculate differences between adjacent frames: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probabilities[:, 1:, :] - probabilities[:, :-1, :]

        # Squared difference
        sq_diff = diff**2

        # Apply truncation (clamping)
        # If threshold is large (e.g. 4.0), this effectively acts as standard MSE (unclamped)
        # since probabilities are in [0, 1], max sq_diff is 1.0.
        if self.threshold is not None:
            sq_diff = torch.clamp(sq_diff, min=0, max=self.threshold)

        # Sum over classes
        # Shape: (B, T-1)
        mse_per_frame = torch.sum(sq_diff, dim=2)

        # Create transition mask: valid if both t and t-1 are valid
        # Shape: (B, T-1)
        mask_trans = mask[:, 1:] * mask[:, :-1]

        # Apply mask
        masked_mse = mse_per_frame * mask_trans

        # Average over valid transitions
        valid_transitions = torch.sum(mask_trans)

        if valid_transitions > 0:
            return torch.sum(masked_mse) / valid_transitions
        else:
            return torch.tensor(0.0, device=probabilities.device)


class HierarchicalLoss(nn.Module):
    """
    Composite loss module for the HG-GCRCN model.
    Aggregates Classification, Boundary, Foreground, and Smoothing losses across all stages.
    """

    def __init__(self):
        super(HierarchicalLoss, self).__init__()

        # Hyperparameters
        self.w_cls = config.HYPERPARAMS["weight_cls"]
        self.w_bnd = config.HYPERPARAMS["weight_bnd"]
        self.w_fg = config.HYPERPARAMS["weight_fg"]
        self.w_smooth = config.HYPERPARAMS["weight_smooth"]

        # Class Weights for Imbalanced Classification
        # Register as buffer to ensure it moves to device with the module
        class_weights = torch.tensor(
            config.HYPERPARAMS["class_weights"], dtype=torch.float32
        )

        # Loss Functions
        # reduction='none' allows manual masking
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.tmse_loss = TMSELoss(threshold=config.HYPERPARAMS["tmse_threshold"])

    def forward(self, stage_outputs, targets):
        """
        Args:
            stage_outputs: List of dictionaries, one for each stage.
                           Each dict contains:
                           - 'cls': (B, T, C) Logits for classification
                           - 'bnd': (B, T, 1) Logits for boundary detection
                           - 'fg':  (B, T, 1) Logits for foreground detection
            targets: Dictionary containing:
                     - 'cls': (B, T) Class indices
                     - 'bnd': (B, T) Boundary labels (0 or 1)
                     - 'fg':  (B, T) Foreground labels (0 or 1)
                     - 'mask': (B, T) Valid frame mask
        Returns:
            total_loss: Scalar tensor
            metrics: Dictionary of loss components
        """
        cls_target = targets["cls"]
        bnd_target = targets["bnd"]
        fg_target = targets["fg"]
        mask = targets["mask"]

        total_loss = 0.0
        metrics = {}

        # Number of valid frames in the batch
        valid_steps = mask.sum()

        if valid_steps == 0:
            # Handle edge case of empty batch (though unlikely with proper collate)
            return torch.tensor(0.0, device=mask.device, requires_grad=True), {}

        # Iterate over stages (Stage 1, Stage 2, Stage 3)
        for i, stage_out in enumerate(stage_outputs):
            stage_name = f"s{i+1}"

            # --- 1. Classification Loss (Weighted Cross Entropy) ---
            cls_logits = stage_out["cls"]  # (B, T, C)
            B, T, C = cls_logits.shape

            # Flatten for CrossEntropyLoss: (B*T, C) vs (B*T)
            loss_cls_raw = self.ce_loss(cls_logits.view(-1, C), cls_target.view(-1))
            loss_cls_raw = loss_cls_raw.view(B, T)

            # Apply mask and average
            loss_cls = (loss_cls_raw * mask).sum() / valid_steps

            # --- 2. Boundary Loss (Binary Cross Entropy) ---
            bnd_logits = stage_out["bnd"].squeeze(-1)  # (B, T)
            loss_bnd_raw = self.bce_loss(bnd_logits, bnd_target)
            loss_bnd = (loss_bnd_raw * mask).sum() / valid_steps

            # --- 3. Foreground Loss (Binary Cross Entropy) ---
            fg_logits = stage_out["fg"].squeeze(-1)  # (B, T)
            loss_fg_raw = self.bce_loss(fg_logits, fg_target)
            loss_fg = (loss_fg_raw * mask).sum() / valid_steps

            # --- 4. Smoothing Loss (T-MSE on Probabilities) ---
            # Convert logits to probabilities
            probs = F.softmax(cls_logits, dim=2)
            loss_smooth = self.tmse_loss(probs, mask)

            # --- Aggregate Stage Loss ---
            stage_loss = (
                self.w_cls * loss_cls
                + self.w_bnd * loss_bnd
                + self.w_fg * loss_fg
                + self.w_smooth * loss_smooth
            )

            total_loss += stage_loss

            # --- Logging ---
            metrics[f"{stage_name}_loss"] = stage_loss.item()
            metrics[f"{stage_name}_cls"] = loss_cls.item()
            metrics[f"{stage_name}_bnd"] = loss_bnd.item()
            metrics[f"{stage_name}_fg"] = loss_fg.item()
            metrics[f"{stage_name}_smooth"] = loss_smooth.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
