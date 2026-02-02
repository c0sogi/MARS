import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Minimizes distance for positive pairs (label=1) and maximizes distance
    for negative pairs (label=0) up to a margin.
    """

    def __init__(self, margin=Config.MARGIN):
        """
        Args:
            margin (float): The margin distance for negative pairs.
                            Pairs with distance > margin will not contribute to loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Computes the contrastive loss.

        Args:
            output1 (torch.Tensor): Embeddings for the first set of images. Shape: (Batch, Embedding_Dim)
            output2 (torch.Tensor): Embeddings for the second set of images. Shape: (Batch, Embedding_Dim)
            label (torch.Tensor): Binary labels. 1.0 for same class, 0.0 for different class. Shape: (Batch,)

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch).
        """
        # Calculate Euclidean distance between the pair of embeddings
        # F.pairwise_distance computes the norm of the difference vector
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Formula:
        # Loss = mean( Y * D^2 + (1 - Y) * max(0, margin - D)^2 )
        # where Y is the label (1 for same, 0 for different) and D is the distance.

        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
