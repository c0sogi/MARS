import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class SubCenterArcFaceLoss(nn.Module):
    """
    Sub-Center ArcFace Loss implementation.

    Reference: "Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces"
    It generalizes ArcFace by allowing k sub-centers for each class, which helps in
    handling intra-class variance (e.g., different views/rooms of a hotel).
    """

    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        embedding_size: int = Config.EMBEDDING_SIZE,
        margin: float = Config.MARGIN,
        scale: float = Config.SCALE,
        k: int = Config.K_SUB_CENTERS,
    ):
        super(SubCenterArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.k = k

        # Weight shape: (num_classes * k, embedding_size)
        # We flatten the class and k dimensions for the linear layer
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * k, embedding_size))

        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Threshold for numerical stability
        # cos(pi - m) = -cos(m)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (Batch_Size, Embedding_Size) - Normalized or unnormalized features
            labels: (Batch_Size,) - Ground truth class indices

        Returns:
            loss: Scalar tensor representing the Cross Entropy Loss
        """
        # 1. Normalize inputs and weights
        # L2 Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # L2 Normalize weights
        weights = F.normalize(self.weight, p=2, dim=1)

        # 2. Compute Cosine Similarity
        # Shape: (Batch_Size, Num_Classes * K)
        cosine_all = F.linear(embeddings, weights)

        # 3. Handle Sub-Centers
        # Reshape to (Batch_Size, Num_Classes, K)
        cosine_all = cosine_all.view(-1, self.num_classes, self.k)
        # Take max over K to get the best sub-center for each class
        # Shape: (Batch_Size, Num_Classes)
        cosine, _ = torch.max(cosine_all, dim=2)

        # 4. Apply ArcFace Margin to the target class logits
        # Clamp for numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Get the cosine value of the ground truth class
        # one_hot = torch.zeros_like(cosine)
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # But we only need to modify the specific indices, so we can do it efficiently

        # Create sine from cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle cases where theta + m > pi (numerical stability check)
        # If cosine > th, use phi. Else, use cosine - mm (Taylor expansion approx or fallback)
        # This prevents gradients from exploding or vanishing in edge cases
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert labels to one-hot encoding to select which logits to modify
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Final logits: Use phi for target class, cosine for others
        # logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # Ideally: output = s * (one_hot * phi + (1 - one_hot) * cosine)
        # Mathematically equivalent to: output = s * cosine + s * one_hot * (phi - cosine)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # 5. Compute Cross Entropy Loss
        loss = F.cross_entropy(output, labels)

        return loss
