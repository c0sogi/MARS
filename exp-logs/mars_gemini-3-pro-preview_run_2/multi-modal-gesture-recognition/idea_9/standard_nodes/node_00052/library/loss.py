import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) Loss.

    This loss function penalizes rapid changes in prediction probabilities between
    consecutive frames to enforce temporal smoothness. It is designed to operate
    on Softmax probabilities (not logits) as per the MD-CRCN architecture specification.

    The loss is computed as:
        L = mean( clamp( sum((P_t - P_{t-1})^2), max=threshold ) )

    It strictly respects the sequence mask, ensuring that transitions involving
    padding frames do not contribute to the loss.
    """

    def __init__(self, threshold=Config.TMSE_THRESHOLD):
        """
        Args:
            threshold (float): The maximum value for the squared difference error.
                               Defaults to Config.TMSE_THRESHOLD.
        """
        super(TMSELoss, self).__init__()
        self.threshold = float(threshold)

    def forward(self, probs, mask):
        """
        Computes the T-MSE loss.

        Args:
            probs (torch.Tensor): Softmax probabilities of shape (Batch, Frames, Classes).
            mask (torch.Tensor): Binary sequence mask of shape (Batch, Frames).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Validate input dimensions
        if probs.dim() != 3:
            raise ValueError(
                f"Expected probs to have 3 dimensions (Batch, Frames, Classes), got {probs.shape}"
            )

        # Calculate temporal difference: P_t - P_{t-1}
        # Slice to align t (1 to T) and t-1 (0 to T-1)
        # Shape: (Batch, Frames-1, Classes)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Calculate squared difference summed over classes
        # Shape: (Batch, Frames-1)
        sq_diff = torch.sum(diff**2, dim=2)

        # Truncate (Clamp) the error to prevent outliers from dominating gradients
        truncated_diff = torch.clamp(sq_diff, max=self.threshold)

        # Create a transition mask
        # A transition is valid only if both frame t and frame t-1 are valid (non-padding)
        # Shape: (Batch, Frames-1)
        transition_mask = mask[:, 1:] * mask[:, :-1]

        # Apply mask to the loss
        masked_loss = truncated_diff * transition_mask

        # Normalize by the number of valid transitions
        # Add a small epsilon to avoid division by zero for empty/single-frame sequences
        loss = masked_loss.sum() / (transition_mask.sum() + 1e-8)

        return loss


class MaskedWeightedCrossEntropy(nn.Module):
    """
    Masked Weighted Cross Entropy Loss.

    This loss function wraps PyTorch's CrossEntropyLoss to add two critical features
    for the MD-CRCN task:
    1. Class Weighting: Applies specific weights to penalize background errors less
       than gesture errors, addressing class imbalance.
    2. Sequence Masking: Strictly ignores padded regions of the input sequences
       during loss calculation.
    """

    def __init__(self):
        super(MaskedWeightedCrossEntropy, self).__init__()

        # Load class weights from Config
        # Weights are registered as a buffer so they are automatically moved to the
        # correct device (CPU/GPU) along with the model.
        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("weight", weights)

    def forward(self, logits, targets, mask):
        """
        Computes the Masked Weighted Cross Entropy loss.

        Args:
            logits (torch.Tensor): Raw logits of shape (Batch, Frames, Classes).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Frames).
            mask (torch.Tensor): Binary sequence mask of shape (Batch, Frames).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # PyTorch CrossEntropyLoss expects inputs in shape (Batch, Classes, Frames)
        # If input is (Batch, Frames, Classes), we permute it.
        if (
            logits.size(1) != Config.NUM_CLASSES
            and logits.size(2) == Config.NUM_CLASSES
        ):
            logits = logits.permute(0, 2, 1)

        # Compute element-wise Cross Entropy Loss
        # reduction='none' allows us to access loss per frame before averaging
        # logits: (B, C, T), targets: (B, T) -> raw_loss: (B, T)
        raw_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none", ignore_index=-1
        )

        # Apply the sequence mask to zero out loss from padding frames
        masked_loss = raw_loss * mask

        # Compute the mean loss over valid frames only
        loss = masked_loss.sum() / (mask.sum() + 1e-8)

        return loss
