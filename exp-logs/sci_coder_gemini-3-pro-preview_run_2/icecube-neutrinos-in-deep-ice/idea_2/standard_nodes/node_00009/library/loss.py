import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    """
    Loss function based on Cosine Similarity.
    Minimizes 1 - cos(theta) between prediction and target.
    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        # eps is used to avoid division by zero in cosine similarity calculation
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, pred, target):
        """
        Computes the loss between predicted vectors and target vectors.

        Args:
            pred (torch.Tensor): Predicted vectors of shape (Batch, 3).
            target (torch.Tensor): Ground truth unit vectors of shape (Batch, 3).

        Returns:
            torch.Tensor: Scalar loss value (mean over batch).
        """
        # Calculate cosine similarity: (A . B) / (|A| * |B|)
        # Output range: [-1, 1]
        cos_sim = self.cosine_sim(pred, target)

        # Loss = 1 - similarity
        # If aligned (sim=1), loss=0.
        # If opposite (sim=-1), loss=2.
        # If orthogonal (sim=0), loss=1.
        loss = 1.0 - cos_sim

        return loss.mean()


def calculate_angular_error(pred, target):
    """
    Calculates the mean angular error in radians between predicted and target vectors.
    This serves as the primary evaluation metric for the task.

    Args:
        pred (torch.Tensor): Predicted vectors of shape (Batch, 3).
        target (torch.Tensor): Ground truth unit vectors of shape (Batch, 3).

    Returns:
        float: Mean angular error in radians.
    """
    with torch.no_grad():
        # Normalize predictions and targets to unit vectors
        # This ensures the dot product corresponds directly to cos(theta)
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Dot product
        dot = torch.sum(pred_norm * target_norm, dim=1)

        # Clamp for numerical stability to avoid NaNs in acos due to float precision errors
        # (e.g., dot product slightly > 1.0)
        dot = torch.clamp(dot, -1.0, 1.0)

        # Calculate angle in radians: arccos(dot)
        angles = torch.acos(dot)

        return angles.mean().item()
