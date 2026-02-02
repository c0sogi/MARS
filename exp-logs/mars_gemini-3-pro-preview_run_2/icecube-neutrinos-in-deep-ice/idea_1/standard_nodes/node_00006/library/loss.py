import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDistanceLoss(nn.Module):
    """
    Cosine Distance Loss for Angular Error Minimization.

    Computes the loss as 1 - CosineSimilarity(predicted_vector, true_vector).
    This acts as a proxy for minimizing the angular error between the predicted
    direction and the true neutrino origin.
    """

    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, pred_vector, target_angles):
        """
        Calculates the cosine distance loss.

        Args:
            pred_vector (torch.Tensor): Predicted 3D vectors of shape (Batch, 3).
                                        These do not need to be normalized beforehand.
            target_angles (torch.Tensor): Ground truth angles of shape (Batch, 2)
                                          containing [azimuth, zenith] in radians.

        Returns:
            torch.Tensor: Scalar loss value representing the mean cosine distance
                          (1 - mean_cosine_similarity) over the batch.
        """
        # 1. Normalize predicted vector to unit length
        # This ensures the vector lies on the unit sphere for valid cosine calculation.
        # eps=1e-12 is default in F.normalize to prevent division by zero.
        pred_norm = F.normalize(pred_vector, p=2, dim=1)

        # 2. Convert ground truth spherical coordinates to Cartesian unit vectors
        # target_angles: [azimuth, zenith]
        azimuth = target_angles[:, 0]
        zenith = target_angles[:, 1]

        # Spherical to Cartesian conversion formulas:
        # x = cos(azimuth) * sin(zenith)
        # y = sin(azimuth) * sin(zenith)
        # z = cos(zenith)
        sin_zenith = torch.sin(zenith)
        true_x = torch.cos(azimuth) * sin_zenith
        true_y = torch.sin(azimuth) * sin_zenith
        true_z = torch.cos(zenith)

        # Stack to form true unit vectors: (Batch, 3)
        true_vector = torch.stack([true_x, true_y, true_z], dim=1)

        # 3. Calculate Cosine Similarity
        # The dot product of two unit vectors equals the cosine of the angle between them.
        # shape: (Batch,)
        cosine_sim = torch.sum(pred_norm * true_vector, dim=1)

        # 4. Compute Loss
        # We want to maximize cosine_sim (make it close to 1).
        # Loss = 1 - cosine_sim
        # This ranges from 0 (perfect alignment) to 2 (opposing directions).
        loss = 1.0 - torch.mean(cosine_sim)

        return loss
