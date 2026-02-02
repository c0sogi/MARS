import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class WhaleEmbeddingNet(nn.Module):
    """
    Embedding Network architecture using a DenseNet-121 backbone.
    Cite solution_lesson_node_00004: Switch to DenseNet121 for better feature extraction.
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

        # Load pre-trained DenseNet-121
        self.backbone = models.densenet121(weights="DEFAULT")

        # Get the number of input features for the final classifier layer
        num_ftrs = self.backbone.classifier.in_features

        # Replace the final classifier layer
        self.backbone.classifier = nn.Linear(num_ftrs, embedding_dim)

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


# ContrastiveLoss removed in favor of nn.TripletMarginLoss used directly in engine.py
# Cite solution_lesson_node_00004: Relative Ranking Objectives (Triplet Loss)
