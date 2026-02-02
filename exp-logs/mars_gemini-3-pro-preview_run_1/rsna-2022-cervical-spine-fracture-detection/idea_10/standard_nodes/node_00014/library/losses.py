import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceBCELoss(nn.Module):
    """
    Loss function for Stage 1 Segmentation.
    Combines Binary Cross Entropy (BCE) and Dice Loss.

    BCE provides smooth gradients for pixel-wise classification.
    Dice Loss handles the class imbalance between background and bone pixels.
    """

    def __init__(self, smooth: float = 1e-6, bce_weight: float = 0.5):
        """
        Args:
            smooth (float): Smoothing factor for Dice calculation to avoid division by zero.
            bce_weight (float): Weight assigned to BCE loss. Dice weight will be (1 - bce_weight).
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W) or (B, H, W).
                                    Values should be 0 or 1.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Binary Cross Entropy
        # inputs are logits, so we use binary_cross_entropy_with_logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="mean"
        )

        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_sigmoid.sum() + targets.sum() + self.smooth
        )
        dice_loss = 1 - dice

        # Weighted combination
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)

        return total_loss


class UnweightedBCELoss(nn.Module):
    """
    Loss function for Stage 2 Slice-Level Encoder.
    Standard Binary Cross Entropy without class balancing or competition weights.

    This forces the encoder to learn robust visual features based on fracture evidence
    rather than optimizing for the skewed metric weights at the local level.
    """

    def __init__(self):
        super(UnweightedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Model logits.
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        return self.bce(inputs, targets.float())


class CompetitionWeightedLoss(nn.Module):
    """
    Loss function for Stage 3 Patient-Level Aggregator.
    Implements the Weighted Multi-label Logarithmic Loss used in the competition.

    Formula: L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    The loss is averaged across all samples and all classes.
    """

    def __init__(self):
        super(CompetitionWeightedLoss, self).__init__()
        # Load weights from Config
        # Config.LOSS_WEIGHTS is typically [1, 1, 1, 1, 1, 1, 1, 7] for C1-C7, patient_overall
        self.weights = Config.LOSS_WEIGHTS

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Model logits of shape (B, 8).
                                   Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
            targets (torch.Tensor): Ground truth labels of shape (B, 8).

        Returns:
            torch.Tensor: Scalar weighted log loss.
        """
        # Ensure weights are on the correct device
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)

        # F.binary_cross_entropy_with_logits with 'weight' argument applies the weight
        # to the loss of each batch element.
        # By passing a weight tensor of shape (8,), it broadcasts to (B, 8).
        # This effectively implements: w_j * BCE(p_ij, y_ij)

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), weight=self.weights, reduction="mean"
        )

        return loss
