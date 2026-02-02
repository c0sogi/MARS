import torch
import torch.nn as nn
from library.config import Config


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for binary/multi-label segmentation.
    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten tensors to calculate global Dice
        # (Batch, Channels, Height, Width) -> (N,)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class WeightedDeepSupervisionLoss(nn.Module):
    """
    Loss function that combines Binary Cross Entropy (BCE) and Dice Loss.
    Supports Weighted Deep Supervision by aggregating losses from multiple decoder heads.
    """

    def __init__(self):
        super(WeightedDeepSupervisionLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.weights = Config.DS_WEIGHTS

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor or list/tuple of torch.Tensor):
                Model output. If list, assumes [final_head, aux_head1, ...].
            targets (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Handle Deep Supervision (List of outputs)
        if isinstance(preds, (list, tuple)):
            total_loss = 0.0

            # Ensure we iterate over available predictions and weights
            # The model returns [final, aux1, aux2, aux3] matching DS_WEIGHTS
            for pred, weight in zip(preds, self.weights):
                bce_loss = self.bce(pred, targets)
                dice_loss = self.dice(pred, targets)

                # Combine BCE and Dice
                head_loss = bce_loss + dice_loss

                # Aggregate weighted loss
                total_loss += weight * head_loss

            return total_loss

        # Handle Single Output (Validation/Inference or Deep Supervision Disabled)
        else:
            bce_loss = self.bce(preds, targets)
            dice_loss = self.dice(preds, targets)
            return bce_loss + dice_loss
