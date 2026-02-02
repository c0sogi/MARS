import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        """
        Combined Binary Cross Entropy and Soft Dice Loss for segmentation tasks.

        The loss is calculated as:
            Loss = bce_weight * BCE + dice_weight * (1 - Dice)

        Args:
            bce_weight (float): Weight coefficient for the Binary Cross Entropy loss.
            dice_weight (float): Weight coefficient for the Dice loss.
            smooth (float): Smoothing factor for the Dice coefficient to prevent
                            division by zero and smooth the gradient.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass to calculate the combined loss.

        Args:
            inputs (torch.Tensor): The raw logits output from the model (before sigmoid).
                                   Shape: (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): The ground truth binary masks.
                                    Shape: Must match inputs or be broadcastable.
                                    Values should be 0 or 1.

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # Ensure targets are the same float type as inputs for calculation
        if targets.dtype != inputs.dtype:
            targets = targets.type_as(inputs)

        # 1. Binary Cross Entropy Loss
        # F.binary_cross_entropy_with_logits applies sigmoid internally and is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # 2. Soft Dice Loss
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to compute the Dice score over the entire batch (or image)
        # Flattening (N, C, H, W) -> (N*C*H*W) treats the batch as a single volume,
        # which is a common strategy. Alternatively, one could compute per image.
        # Here we flatten entirely to match the behavior of standard implementations.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice coefficient: (2 * |A n B| + smooth) / (|A| + |B| + smooth)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice score
        dice_loss = 1.0 - dice_score

        # 3. Combined Loss
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
