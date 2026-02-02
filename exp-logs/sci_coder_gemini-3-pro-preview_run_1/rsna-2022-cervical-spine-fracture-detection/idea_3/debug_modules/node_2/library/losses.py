import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedLogLoss(nn.Module):
    """
    Competition-specific Weighted Multi-label Logarithmic Loss.

    Formula:
        L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]

    Weights:
        patient_overall: 7.0
        C1-C7: 1.0

    The loss is averaged across all rows (samples * classes).
    """

    def __init__(self):
        super(WeightedLogLoss, self).__init__()
        # Define weights corresponding to columns: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        weights = torch.tensor(
            [Config.WEIGHT_PATIENT_OVERALL] + [Config.WEIGHT_VERTEBRAE] * 7,
            dtype=torch.float32,
        )
        # Register as buffer to automatically move with model to device (CPU/GPU)
        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (Batch, 8).
            targets (torch.Tensor): Ground truth labels (Batch, 8).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Clamp probabilities to avoid log(0)
        epsilon = 1e-7
        probs = torch.clamp(probs, epsilon, 1.0 - epsilon)

        # Calculate binary cross entropy terms per element
        # Note: We implement this manually to apply class-specific weights easily
        # before reduction.
        loss_terms = -(
            targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)
        )

        # Apply weights to each class
        weighted_loss = loss_terms * self.weights

        # Average over all entries (Batch Size * Num Classes)
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for Segmentation (Stage 1 Localizer).
    Optimizes the overlap between predicted and ground truth masks.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (Batch, 1, H, W)
            targets (torch.Tensor): (Batch, 1, H, W)
        """
        # Apply sigmoid to get probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Calculate Dice Coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class EncoderLoss(nn.Module):
    """
    Loss for Stage 2 Slice Encoder.
    Uses standard Binary Cross Entropy with Logits.

    This stage is trained to detect fracture features in individual slices
    without the competition-specific weighting, to encourage high precision.
    """

    def __init__(self):
        super(EncoderLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (Batch, 1) or (Batch, N)
            targets (torch.Tensor): (Batch, 1) or (Batch, N)
        """
        return self.loss_fn(logits, targets)
