import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Module.

    This module acts as the metric learning head for the model. It projects
    input embeddings onto a hypersphere, calculates cosine similarities with
    learnable class centers, applies an additive angular margin to the target
    classes, and computes the Cross Entropy loss.

    Reference: Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features=None, out_features=None, s=None, m=None):
        """
        Args:
            in_features (int): Size of input embeddings. Defaults to Config.MODEL_CONFIGS[0]['embedding_size'].
            out_features (int): Number of classes. Defaults to Config.N_CLASSES.
            s (float): Norm scale parameter. Defaults to Config.ARCFACE_SCALE.
            m (float): Angular margin parameter. Defaults to Config.ARCFACE_MARGIN.
        """
        super(ArcFaceLoss, self).__init__()

        # Set parameters with fallbacks to Config
        self.in_features = (
            in_features
            if in_features is not None
            else Config.MODEL_CONFIGS[0]["embedding_size"]
        )
        self.out_features = (
            out_features if out_features is not None else Config.N_CLASSES
        )
        self.s = s if s is not None else Config.ARCFACE_SCALE
        self.m = m if m is not None else Config.ARCFACE_MARGIN

        # Learnable weights (Class Centers)
        # Shape: (Num_Classes, Embedding_Size)
        self.weight = nn.Parameter(
            torch.FloatTensor(self.out_features, self.in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants for the margin function
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # Threshold for "easy margin" / stability logic
        # If theta + m > pi, the function is not monotonic. We use a fallback.
        # th = cos(pi - m)
        self.th = math.cos(math.pi - self.m)
        # mm = sin(pi - m) * m = sin(m) * m
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings, labels):
        """
        Computes the ArcFace loss.

        Args:
            embeddings (torch.Tensor): Raw feature vectors from the backbone. Shape (Batch, Embedding_Size).
            labels (torch.Tensor): Ground truth class indices. Shape (Batch,).

        Returns:
            torch.Tensor: Scalar CrossEntropy loss.
        """
        # 1. Normalize Inputs and Weights
        # L2 normalization ensures we are working on the hypersphere
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weights_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. Compute Cosine Similarity (Logits)
        # Shape: (Batch, Num_Classes)
        cosine = F.linear(embeddings_norm, weights_norm)

        # 3. Apply Additive Angular Margin to Targets
        # We only modify the logits corresponding to the true class labels.

        # Clamp cosine values for numerical stability in acos
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Gather the cosine values of the target classes
        # labels shape: (B) -> (B, 1) for gather compatibility
        index = labels.view(-1, 1)
        cosine_target = cosine.gather(1, index)

        # Calculate cos(theta + m) using trigonometric identities:
        # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        sin_theta = torch.sqrt(1.0 - torch.pow(cosine_target, 2))
        cosine_phi = cosine_target * self.cos_m - sin_theta * self.sin_m

        # Handle stability where theta + m > pi
        # If cos(theta) > cos(pi - m), then theta < pi - m, so theta + m < pi (Safe).
        # Otherwise, use a penalized fallback (cosine - margin_penalty) to ensure gradients.
        cond = cosine_target > self.th
        cosine_target_margin = torch.where(cond, cosine_phi, cosine_target - self.mm)

        # 4. Update the Logits
        # Scatter the modified target logits back into the cosine matrix
        # We use scatter (not in-place) to preserve gradients properly
        output = cosine.scatter(1, index, cosine_target_margin)

        # 5. Scale the Logits
        output = output * self.s

        # 6. Compute Cross Entropy Loss
        loss = F.cross_entropy(output, labels)

        return loss
