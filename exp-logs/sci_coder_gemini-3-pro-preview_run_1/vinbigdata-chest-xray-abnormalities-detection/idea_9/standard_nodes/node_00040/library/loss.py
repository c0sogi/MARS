import torch
import torch.nn as nn
import torch.nn.functional as F


class ThoracicLoss(nn.Module):
    """
    Multi-task loss function for the Thoracic Disease Detection model.

    Components:
    1. Modified Focal Loss (CenterNet style) for Heatmap Classification.
    2. Masked L1 Loss for Bounding Box Size Regression.
    3. Masked L1 Loss for Sub-pixel Offset Regression.
    4. BCEWithLogitsLoss for Global 'Finding vs No Finding' Classification.
    """

    def __init__(
        self, weight_hm=1.0, weight_size=0.1, weight_offset=1.0, weight_global=0.5
    ):
        """
        Args:
            weight_hm (float): Weight for the heatmap focal loss.
            weight_size (float): Weight for the size regression loss.
            weight_offset (float): Weight for the offset regression loss.
            weight_global (float): Weight for the global classification loss.
        """
        super().__init__()
        self.weight_hm = weight_hm
        self.weight_size = weight_size
        self.weight_offset = weight_offset
        self.weight_global = weight_global
        self.bce = nn.BCEWithLogitsLoss()

    def focal_loss(self, pred_logits, gt):
        """
        Modified focal loss for heatmap regression (from CenterNet).

        Args:
            pred_logits (torch.Tensor): Predicted heatmap logits (B, C, H, W).
            gt (torch.Tensor): Ground truth heatmap [0, 1] (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight negative examples near the center less (Gaussian penalty)
        neg_weights = torch.pow(1 - gt, 4)

        # Sigmoid activation with numerical stability clamping
        pred = torch.clamp(torch.sigmoid(pred_logits), 1e-6, 1 - 1e-6)

        # Loss calculation
        # Positive cases: (1 - pred)^2 * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

        # Negative cases: (1 - gt)^4 * pred^2 * log(1 - pred)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def reg_l1_loss(self, pred, target, mask):
        """
        L1 Regression loss masked by object presence.

        Args:
            pred (torch.Tensor): Predicted values (B, 2, H, W).
            target (torch.Tensor): Ground truth values (B, 2, H, W).
            mask (torch.Tensor): Object presence mask (B, 1, H, W) or (B, 2, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Expand mask if necessary to match prediction channels (e.g., 2 for x,y or w,h)
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand_as(pred)

        loss = F.l1_loss(pred, target, reduction="none")
        loss = loss * mask

        # Normalize by number of positive pixels
        num_pos = mask.sum() + 1e-4
        return loss.sum() / num_pos

    def forward(self, outputs, targets):
        """
        Calculates the weighted sum of all losses.

        Args:
            outputs (dict): Model outputs containing 'heatmap', 'size', 'offset', 'global_logits'.
            targets (dict): Ground truth containing 'heatmap', 'size', 'offset', 'mask', 'global_label'.

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # 1. Heatmap Loss (Focal)
        hm_loss = self.focal_loss(outputs["heatmap"], targets["heatmap"])

        # 2. Regression Losses (Masked L1)
        # Size Head: Predicts Width and Height
        size_loss = self.reg_l1_loss(outputs["size"], targets["size"], targets["mask"])

        # Offset Head: Predicts sub-pixel discretization error
        offset_loss = self.reg_l1_loss(
            outputs["offset"], targets["offset"], targets["mask"]
        )

        # 3. Global Classification Loss (BCE)
        # Used as a gate for inference
        global_loss = self.bce(outputs["global_logits"], targets["global_label"])

        # Weighted Sum
        total_loss = (
            self.weight_hm * hm_loss
            + self.weight_size * size_loss
            + self.weight_offset * offset_loss
            + self.weight_global * global_loss
        )

        metrics = {
            "hm_loss": hm_loss.item(),
            "size_loss": size_loss.item(),
            "off_loss": offset_loss.item(),
            "glob_loss": global_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
