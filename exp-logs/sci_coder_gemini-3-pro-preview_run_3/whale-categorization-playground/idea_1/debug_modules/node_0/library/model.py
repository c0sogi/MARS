import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class WhaleEmbeddingNet(nn.Module):
    """
    Siamese Network architecture using a ResNet-18 backbone.
    The final classification layer is replaced to output a fixed-size embedding.
    """

    def __init__(self, embedding_dim=None):
        """
        Args:
            embedding_dim (int, optional): Size of the output embedding vector.
                                           Defaults to Config.EMBEDDING_DIM.
        """
        super(WhaleEmbeddingNet, self).__init__()

        if embedding_dim is None:
            embedding_dim = Config.EMBEDDING_DIM

        # Load pre-trained ResNet-18
        # Using weights='DEFAULT' which corresponds to the best available pre-trained weights
        self.backbone = models.resnet18(weights="DEFAULT")

        # Get the number of input features for the final FC layer
        num_ftrs = self.backbone.fc.in_features

        # Replace the final fully connected layer
        # This projects the high-dimensional features to the embedding space
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        """
        Forward pass to generate embeddings.

        Args:
            x (torch.Tensor): Input images tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Embedding vectors of shape (Batch, Embedding_Dim).
        """
        # ResNet forward pass (includes conv layers, avg pool, and the modified fc)
        output = self.backbone(x)

        # Optional: Normalize embeddings to unit length?
        # While common in some setups (like ArcFace), standard Contrastive Loss
        # with Euclidean distance often works on raw unnormalized vectors or
        # vectors normalized afterwards. We return the raw linear output here.
        return output

    def get_embedding(self, x):
        """
        Helper method to get embedding (alias for forward, useful for inference semantics).
        """
        return self.forward(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Based on: "Dimensionality Reduction by Learning an Invariant Mapping" (Hadsell et al., CVPR 2006).

    Formula:
        L = 0.5 * (Y * D^2 + (1 - Y) * max(0, margin - D)^2)

    Where:
        Y = 1 if inputs are a positive pair (same whale)
        Y = 0 if inputs are a negative pair (different whale)
        D = Euclidean distance between embeddings
    """

    def __init__(self, margin=None):
        """
        Args:
            margin (float, optional): Minimum distance margin for negative pairs.
                                      Defaults to Config.MARGIN.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin if margin is not None else Config.MARGIN

    def forward(self, output1, output2, target):
        """
        Computes the contrastive loss.

        Args:
            output1 (torch.Tensor): Embeddings for the first image in the pair. Shape (Batch, Dim).
            output2 (torch.Tensor): Embeddings for the second image in the pair. Shape (Batch, Dim).
            target (torch.Tensor): Binary labels. 1 for same class, 0 for different. Shape (Batch,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Euclidean distance between the two embeddings
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Ensure target is the correct shape for broadcasting
        # If target is (Batch,), make it (Batch, 1)
        if target.dim() == 1:
            target = target.view(-1, 1)

        # Calculate loss components
        # Component 1: Positive pairs (target == 1) -> Minimize distance
        loss_contrastive = torch.mean(
            (target) * torch.pow(euclidean_distance, 2)
            + (1 - target)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        # Usually multiplied by 0.5, but can be omitted if learning rate is adjusted.
        # We include it to stick to the standard definition.
        return 0.5 * loss_contrastive
