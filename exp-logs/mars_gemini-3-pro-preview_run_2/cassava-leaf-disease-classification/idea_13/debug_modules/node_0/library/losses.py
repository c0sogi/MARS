import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy


class CassavaLoss(nn.Module):
    """
    Custom Loss function for Cassava Leaf Disease Classification.

    Dynamically switches between:
    1. SoftTargetCrossEntropy: When targets are one-hot/soft probabilities (e.g., MixUp/CutMix active).
    2. LabelSmoothingCrossEntropy: When targets are hard integer labels (e.g., Fine-tuning phase).

    This logic relies on the shape of the target tensor:
    - If target.ndim == input.ndim (Batch, NumClasses), use SoftTargetCrossEntropy.
    - If target.ndim == input.ndim - 1 (Batch,), use LabelSmoothingCrossEntropy.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing (float): The label smoothing factor to use when targets are hard labels.
                               Default is 0.1 (standard for Phase 2).
        """
        super(CassavaLoss, self).__init__()
        self.smoothing = smoothing

        # Loss for mixed/soft targets (Phase 1)
        self.soft_target_loss = SoftTargetCrossEntropy()

        # Loss for hard targets with smoothing (Phase 2)
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            input (torch.Tensor): Model predictions (logits) of shape (B, C).
            target (torch.Tensor): Ground truth.
                                   Shape (B, C) for soft targets.
                                   Shape (B,) for hard targets.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Check if targets are soft (probabilities) or hard (indices)
        if target.ndim == input.ndim:
            # Phase 1: MixUp/CutMix active, targets are probabilities
            return self.soft_target_loss(input, target)
        else:
            # Phase 2: No MixUp, targets are class indices
            return self.label_smoothing_loss(input, target)
