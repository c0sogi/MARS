import torch
import torch.nn as nn
from library.config import Config


class GlobalBatchDiceLoss(nn.Module):
    """
    Computes the Global Dice Coefficient Loss for a batch of predictions.

    This loss treats the entire batch as a single volume, flattening all samples
    into one large vector. This stabilizes the gradient and metric calculation,
    especially when many samples in the batch are empty (common in contrail detection).

    Formula:
        Loss = 1 - (2 * Intersection + epsilon) / (Cardinality + epsilon)
        where Intersection and Cardinality are summed over the entire batch.
    """

    def __init__(self, smooth=1e-6):
        super(GlobalBatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities (0-1). Shape (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks (0 or 1). Shape (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten inputs and targets to (N,)
        # This aggregates all pixels from all images in the batch
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        # Cardinality: Sum of probabilities + Sum of true pixels
        cardinality = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_score


class SoftGatedLoss(nn.Module):
    """
    Composite loss function for the Soft-Gated Multi-Task architecture.

    Combines:
    1. Global Batch Dice Loss for the segmentation mask.
    2. Binary Cross Entropy Loss for the classification head (presence/absence).

    The total loss is: L_total = L_Dice + lambda * L_BCE
    """

    def __init__(self, lambda_cls=Config.LAMBDA_CLS, smooth=1e-6):
        super(SoftGatedLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.dice_loss_fn = GlobalBatchDiceLoss(smooth=smooth)
        self.bce_loss_fn = nn.BCELoss()

        # storage for logging components if needed by the training loop
        self.last_metrics = {}

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'mask': Tensor of shape (B, 1, H, W), soft-gated probabilities.
                - 'cls': Tensor of shape (B, 1), classification probabilities.
            targets (torch.Tensor): Ground truth masks of shape (B, 1, H, W).

        Returns:
            torch.Tensor: The weighted total loss.
        """
        pred_mask = outputs["mask"]
        pred_cls = outputs["cls"]

        # 1. Segmentation Loss
        # Global Batch Dice Loss on the final soft-gated mask
        loss_seg = self.dice_loss_fn(pred_mask, targets)

        # 2. Classification Loss
        # Derive classification targets from the ground truth masks
        # If a mask has any positive pixels (1), the image is positive (1).
        # We use no_grad because the target generation is deterministic and not part of the graph
        with torch.no_grad():
            # view(B, -1) flattens H,W. sum(1) sums pixels per image.
            # If sum > 0, it's a positive sample.
            target_cls = (
                (targets.view(targets.size(0), -1).sum(dim=1) > 0).float().view(-1, 1)
            )

        loss_cls = self.bce_loss_fn(pred_cls, target_cls)

        # Total Loss
        total_loss = loss_seg + self.lambda_cls * loss_cls

        # Store metrics for logging (detached from graph)
        self.last_metrics = {
            "loss_total": total_loss.item(),
            "loss_seg": loss_seg.item(),
            "loss_cls": loss_cls.item(),
        }

        return total_loss
