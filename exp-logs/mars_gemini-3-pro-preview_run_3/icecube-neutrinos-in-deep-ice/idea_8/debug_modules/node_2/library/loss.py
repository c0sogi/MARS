import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import angles_to_direction


class CosineSimilarityLoss(nn.Module):
    """
    Custom loss function that computes 1 - cosine_similarity(predicted_vector, target_vector).

    This loss directly optimizes the angular alignment between the predicted direction
    and the true neutrino origin. Unlike MSE on angles, this avoids issues with
    periodicity (azimuth 0 vs 2pi) and the singularity at the poles (zenith 0 or pi).
    """

    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, pred, true_azimuth, true_zenith):
        """
        Computes the cosine similarity loss.

        Args:
            pred (torch.Tensor): Predicted vectors of shape (Batch_Size, 3).
                                 These do not need to be normalized beforehand.
            true_azimuth (torch.Tensor): Ground truth azimuth angles in radians. Shape (Batch_Size,).
            true_zenith (torch.Tensor): Ground truth zenith angles in radians. Shape (Batch_Size,).

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # 1. Convert target angles to 3D unit vectors
        # ensure inputs are on the same device
        true_azimuth = true_azimuth.to(pred.device)
        true_zenith = true_zenith.to(pred.device)

        target_vectors = angles_to_direction(true_azimuth, true_zenith)

        # 2. Normalize predicted vectors to ensure they are unit length
        # The network output might not be normalized, so we enforce it here.
        pred_vectors = F.normalize(pred, p=2, dim=1)

        # 3. Compute Cosine Similarity
        # Dot product of two unit vectors is the cosine of the angle between them.
        # shape: (Batch_Size,)
        cos_sim = torch.sum(pred_vectors * target_vectors, dim=1)

        # 4. Compute Loss
        # We want to maximize similarity (close to 1), so we minimize (1 - similarity).
        # Range: [0, 2], where 0 is perfect alignment and 2 is opposite direction.
        loss = 1.0 - cos_sim.mean()

        return loss
