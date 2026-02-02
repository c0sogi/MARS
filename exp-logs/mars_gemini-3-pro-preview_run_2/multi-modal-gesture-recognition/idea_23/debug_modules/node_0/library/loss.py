import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SoftBoundaryLoss(nn.Module):
    """
    Computes Mean Squared Error between predicted boundary probabilities and
    Gaussian soft targets. Used for the boundary regression head.
    """

    def __init__(self):
        super(SoftBoundaryLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, Time, 1) or (Batch, Time).
            targets (torch.Tensor): Gaussian soft targets of shape (Batch, Time).
            mask (torch.Tensor): Boolean mask of shape (Batch, Time) indicating valid frames.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure logits are (B, T)
        if logits.dim() == 3:
            logits = logits.squeeze(-1)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Compute element-wise MSE
        loss = self.mse(probs, targets)

        # Apply mask to zero out padding
        loss = loss * mask

        # Average over valid frames only
        return loss.sum() / mask.sum().clamp(min=1.0)


class TMSELoss(nn.Module):
    """
    Temporal Mean Squared Error (T-MSE) Loss.
    Penalizes rapid changes in class probabilities between adjacent frames to
    encourage temporal smoothness in predictions.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits, mask):
        """
        Args:
            logits (torch.Tensor): Predicted logits of shape (Batch, Time, NumClasses).
            mask (torch.Tensor): Boolean mask of shape (Batch, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to probabilities using Softmax (Unclamped)
        probs = torch.softmax(logits, dim=-1)

        # Calculate difference between probabilities at t and t-1
        p_t = probs[:, 1:, :]
        p_tm1 = probs[:, :-1, :]

        # Sum squared differences over classes: (B, T-1)
        loss = self.mse(p_t, p_tm1).sum(dim=-1)

        # Align mask to T-1 length (valid transitions only)
        mask_t = mask[:, 1:]
        mask_tm1 = mask[:, :-1]
        valid_mask = (mask_t * mask_tm1).float()

        # Apply mask
        loss = loss * valid_mask

        # Average over valid transitions
        return loss.sum() / valid_mask.sum().clamp(min=1.0)


class MultiStageLoss(nn.Module):
    """
    Aggregates losses from all three stages of the SymG-CRCN model.
    For each stage, it computes:
    1. Weighted Cross Entropy Loss for classification.
    2. Soft Boundary MSE Loss for boundary detection.
    3. Temporal MSE Loss for prediction smoothing.
    """

    def __init__(self):
        super(MultiStageLoss, self).__init__()

        # Load class weights from config and register as buffer
        weights = torch.tensor(Config.CLASS_WEIGHTS_LIST, dtype=torch.float)

        # Classification Loss
        # ignore_index=-1 handles the padded labels from collate_fn
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

        # Boundary Regression Loss
        self.bnd_loss = SoftBoundaryLoss()

        # Smoothing Loss
        self.tmse_loss = TMSELoss()

        # Loss Component Weights
        self.w_cls = Config.W_CLS
        self.w_bnd = Config.W_BND
        self.w_smooth = Config.W_SMOOTH

    def forward(self, model_outputs, labels, boundaries, mask):
        """
        Args:
            model_outputs (dict): Dictionary containing logits for stage1, stage2, stage3.
                Keys: 'stageX_cls', 'stageX_bnd'
            labels (torch.Tensor): Ground truth class indices (Batch, Time).
            boundaries (torch.Tensor): Ground truth soft boundary targets (Batch, Time).
            mask (torch.Tensor): Valid frame mask (Batch, Time).

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        total_loss = 0.0
        metrics = {}

        stages = ["stage1", "stage2", "stage3"]

        for stage in stages:
            # Retrieve logits for the current stage
            cls_logits = model_outputs[f"{stage}_cls"]  # Shape: (B, T, C)
            bnd_logits = model_outputs[f"{stage}_bnd"]  # Shape: (B, T, 1)

            # 1. Classification Loss
            # Flatten batch and time dimensions for CrossEntropyLoss
            # (B, T, C) -> (B*T, C) and (B, T) -> (B*T)
            B, T, C = cls_logits.shape
            l_cls = self.ce_loss(cls_logits.reshape(-1, C), labels.reshape(-1))

            # 2. Boundary Loss
            l_bnd = self.bnd_loss(bnd_logits, boundaries, mask)

            # 3. Smoothing Loss
            l_smooth = self.tmse_loss(cls_logits, mask)

            # Weighted Sum for current stage
            stage_loss = (
                (self.w_cls * l_cls) + (self.w_bnd * l_bnd) + (self.w_smooth * l_smooth)
            )

            total_loss += stage_loss

            # Record metrics (detached for logging)
            metrics[f"{stage}_loss"] = stage_loss.item()
            metrics[f"{stage}_cls"] = l_cls.item()
            metrics[f"{stage}_bnd"] = l_bnd.item()
            metrics[f"{stage}_smooth"] = l_smooth.item()

        return total_loss, metrics
