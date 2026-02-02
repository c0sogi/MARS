import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    A hybrid loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    This loss is particularly effective for image segmentation tasks where there is
    a significant class imbalance between foreground (ink) and background (papyrus).

    BCE provides a smooth gradient for pixel-level classification, while Dice Loss
    directly optimizes the overlap metric (F1 score), helping to recover coherent
    shapes.
    """

    def __init__(
        self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6
    ):
        """
        Initialize the BCEDiceLoss.

        Args:
            bce_weight (float): Weight assigned to the BCE component. Defaults to 0.5.
            dice_weight (float): Weight assigned to the Dice component. Defaults to 0.5.
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        # BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce_func = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined loss.

        Args:
            inputs (torch.Tensor): The raw output logits from the model (before sigmoid).
                                   Shape: (Batch, Channels, Height, Width) or (Batch, Height, Width).
            targets (torch.Tensor): The binary ground truth masks.
                                    Shape should match inputs.

        Returns:
            torch.Tensor: The calculated scalar loss.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # --- BCE Component ---
        # inputs are logits, so we use BCEWithLogitsLoss
        bce_loss = self.bce_func(inputs, targets)

        # --- Dice Component ---
        # Apply sigmoid to convert logits to probabilities [0, 1]
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to compute the metric over the entire batch or image
        # This treats the batch as a single large volume for the Dice calculation,
        # or we can view it per sample. Here we flatten all to (N,) for global Dice.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Calculate Dice coefficient
        # Dice = (2 * |A intersect B|) / (|A| + |B|)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice Coefficient
        dice_loss = 1.0 - dice_score

        # --- Combined Loss ---
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss
