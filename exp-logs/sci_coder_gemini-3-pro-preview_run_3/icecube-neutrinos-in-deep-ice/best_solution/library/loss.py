import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import angles_to_direction


class CosineSimilarityLoss(nn.Module):
    """
    Custom loss function for angular regression.
    Optimizes the cosine similarity between predicted and true direction vectors.

    Formula: Loss = 1 - (predicted_vector . true_vector)
    Since vectors are normalized, dot product equals cosine similarity.
    Minimizing this maximizes alignment.
    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted vectors of shape (Batch, 3).
            target (torch.Tensor): Ground truth.
                                   Can be (Batch, 2) containing [azimuth, zenith]
                                   OR (Batch, 3) containing [x, y, z].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Normalize predictions to ensure they are unit vectors
        pred_norm = F.normalize(pred, p=2, dim=1)

        # Handle target format
        if target.shape[1] == 2:
            # Target is (azimuth, zenith)
            azimuth = target[:, 0]
            zenith = target[:, 1]
            target_vec = angles_to_direction(azimuth, zenith)
        elif target.shape[1] == 3:
            # Target is already (x, y, z)
            target_vec = target
            # Ensure target is normalized (ground truth usually is, but for safety)
            target_vec = F.normalize(target_vec, p=2, dim=1)
        else:
            raise ValueError(
                f"Unexpected target shape: {target.shape}. Expected (B, 2) or (B, 3)."
            )

        # Ensure target is on the correct device and dtype
        target_vec = target_vec.to(pred.device).type(pred.dtype)

        # Compute cosine similarity: dot product of unit vectors
        # Shape: (Batch,)
        cosine_sim = torch.sum(pred_norm * target_vec, dim=1)

        # Loss = 1 - mean(cosine_similarity)
        # We want to maximize similarity (close to 1), so we minimize 1 - sim (close to 0)
        loss = 1.0 - torch.mean(cosine_sim)

        return loss


def get_angular_error(pred, target):
    """
    Computes the Mean Angular Error (MAE) in radians between predictions and targets.
    This is the competition metric.

    Args:
        pred (torch.Tensor): Predicted vectors (Batch, 3).
        target (torch.Tensor): Ground truth (Batch, 2) or (Batch, 3).

    Returns:
        float: Mean angular error in radians.
    """
    with torch.no_grad():
        # Normalize predictions
        pred_norm = F.normalize(pred, p=2, dim=1)

        # Process targets
        if target.shape[1] == 2:
            azimuth = target[:, 0]
            zenith = target[:, 1]
            target_vec = angles_to_direction(azimuth, zenith)
        else:
            target_vec = target
            target_vec = F.normalize(target_vec, p=2, dim=1)

        target_vec = target_vec.to(pred.device).type(pred.dtype)

        # Compute cosine similarity
        cosine_sim = torch.sum(pred_norm * target_vec, dim=1)

        # Clamp values to [-1, 1] to avoid NaNs in arccos due to float precision
        # Using a small epsilon buffer
        cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute angles (arccosine)
        angles = torch.acos(cosine_sim)

        # Return mean
        return torch.mean(angles).item()
