import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    A composite loss function combining Binary Cross Entropy (BCE) and Dice Loss.
    This is effective for segmentation tasks, particularly when dealing with class imbalance.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-7):
        """
        Args:
            bce_weight (float): Weight for the BCE component.
            dice_weight (float): Weight for the Dice component.
            smooth (float): Smoothing factor to avoid division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Calculates the combined BCE and Dice loss.

        Args:
            inputs (torch.Tensor): The raw logits from the model (before sigmoid).
                                   Shape: (batch_size, 1, height, width) or similar.
            targets (torch.Tensor): The binary ground truth masks.
                                    Shape: Same as inputs.

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # --- BCE Loss ---
        # BCEWithLogitsLoss handles the sigmoid internally for numerical stability
        bce_loss = self.bce_loss_fn(inputs, targets)

        # --- Dice Loss ---
        # Apply sigmoid to get probabilities for Dice calculation
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to ensure global calculation over the batch or image
        # Using view(-1) flattens the tensor to a 1D vector
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # --- Combined Loss ---
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
