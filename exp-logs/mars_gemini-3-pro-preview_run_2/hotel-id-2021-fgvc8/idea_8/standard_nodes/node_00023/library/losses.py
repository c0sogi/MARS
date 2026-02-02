import torch
import torch.nn as nn


class SubCenterArcFaceLoss(nn.Module):
    """
    Sub-Center ArcFace Loss.

    This module computes the Cross Entropy Loss on the logits produced by the
    SubCenterArcFaceHead. The head in `library/models.py` is responsible for:
    1. Calculating Cosine Similarity between embeddings and K sub-centers.
    2. Selecting the maximum cosine similarity per class (Max-out).
    3. Applying the ArcFace margin penalty (m) to the ground truth class logits.
    4. Scaling the logits by the scale factor (s).

    Therefore, this loss function strictly applies Softmax Cross Entropy to the
    resulting logits to optimize the embedding space.
    """

    def __init__(self, label_smoothing: float = 0.0, reduction: str = "mean"):
        """
        Args:
            label_smoothing (float): Float in [0.0, 1.0]. Specifies the amount of smoothing
                                     when computing the loss. Default is 0.0 as per
                                     strategy requirements to maximize cluster compactness.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super(SubCenterArcFaceLoss, self).__init__()

        # Initialize CrossEntropyLoss.
        # Note: The input logits are already scaled and have the margin applied
        # by the model head during the forward pass with labels.
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction=reduction
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits (torch.Tensor): Predicted logits from the model head.
                                   Shape: (Batch Size, Num Classes).
                                   These logits should already include the margin penalty
                                   for the target class.
            labels (torch.Tensor): Ground truth class indices.
                                   Shape: (Batch Size,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        return self.criterion(logits, labels)
