import torch
import torch.nn as nn
from library.config import Config


class YoloLoss(nn.Module):
    """
    Loss function for the BevYolo 3D Object Detection model.
    Computes a weighted sum of Objectness, Regression, and Classification losses.
    """

    def __init__(self, w_obj=1.0, w_reg=2.0, w_cls=1.0):
        """
        Args:
            w_obj (float): Weight for objectness loss.
            w_reg (float): Weight for regression loss.
            w_cls (float): Weight for classification loss.
        """
        super(YoloLoss, self).__init__()

        # Loss components
        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # MSELoss for coordinate and dimension regression
        self.mse_loss = nn.MSELoss(reduction="none")

        # CrossEntropyLoss for multi-class classification (expects logits)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Weights
        self.w_obj = w_obj
        self.w_reg = w_reg
        self.w_cls = w_cls

    def forward(self, predictions, targets):
        """
        Compute the loss.

        Args:
            predictions: (B, A, H, W, 9 + Num_Classes)
                Tensor containing predicted logits and regression values.
                Channels: [obj_logit, dx, dy, dw, dl, z, dh, sin, cos, class_logits...]
            targets: (B, A, H, W, 10)
                Tensor containing ground truth values.
                Channels: [valid_mask, dx, dy, dw, dl, z, dh, sin, cos, class_idx]

        Returns:
            total_loss (Tensor): Scalar tensor representing the weighted total loss.
            metrics (dict): Dictionary containing individual loss components (float).
        """
        device = predictions.device

        # ==========================
        # 1. Unpack Targets
        # ==========================
        # valid_mask indicates which anchors contain an object (1.0) vs background (0.0)
        # Shape: (B, A, H, W)
        valid_mask = targets[..., 0] == 1.0

        # Regression targets: [dx, dy, dw, dl, z, dh, sin, cos]
        # Shape: (B, A, H, W, 8)
        target_reg = targets[..., 1:9]

        # Classification target: class index
        # Shape: (B, A, H, W)
        target_cls = targets[..., 9].long()

        # ==========================
        # 2. Unpack Predictions
        # ==========================
        # Objectness logit
        pred_obj = predictions[..., 0]

        # Regression predictions
        pred_reg = predictions[..., 1:9]

        # Classification logits
        pred_cls = predictions[..., 9:]

        # ==========================
        # 3. Objectness Loss
        # ==========================
        # Applied to ALL grid cells (both object and background)
        # Target is simply the valid flag (0.0 or 1.0)
        target_obj = targets[..., 0]

        loss_obj = self.bce_loss(pred_obj, target_obj)
        loss_obj = loss_obj.mean()

        # ==========================
        # 4. Regression & Class Loss
        # ==========================
        # Applied ONLY to grid cells containing an object (valid_mask == True)

        num_pos = valid_mask.sum()

        if num_pos > 0:
            # Filter predictions and targets by mask
            p_reg_masked = pred_reg[valid_mask]  # (N_pos, 8)
            t_reg_masked = target_reg[valid_mask]  # (N_pos, 8)

            p_cls_masked = pred_cls[valid_mask]  # (N_pos, Num_Classes)
            t_cls_masked = target_cls[valid_mask]  # (N_pos,)

            # Regression Loss (MSE)
            loss_reg = self.mse_loss(p_reg_masked, t_reg_masked)
            loss_reg = loss_reg.mean()

            # Classification Loss (Cross Entropy)
            loss_cls = self.ce_loss(p_cls_masked, t_cls_masked)
            loss_cls = loss_cls.mean()
        else:
            # Fallback if batch has no objects
            loss_reg = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)

        # ==========================
        # 5. Total Loss
        # ==========================
        total_loss = (
            (self.w_obj * loss_obj) + (self.w_reg * loss_reg) + (self.w_cls * loss_cls)
        )

        # Return metrics for logging
        metrics = {
            "loss_obj": loss_obj.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
