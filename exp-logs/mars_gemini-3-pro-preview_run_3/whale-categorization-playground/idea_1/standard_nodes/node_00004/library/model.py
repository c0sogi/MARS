import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class WhaleEmbeddingNet(nn.Module):
    """
    Triplet Network architecture using a DenseNet-121 backbone.
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

        # Load pre-trained DenseNet-121
        self.backbone = models.densenet121(weights="DEFAULT")

        # Get the number of input features for the final classifier layer
        # DenseNet uses 'classifier' instead of 'fc'
        num_ftrs = self.backbone.classifier.in_features

        # Replace the final fully connected layer
        self.backbone.classifier = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        """
        Forward pass to generate embeddings.
        """
        output = self.backbone(x)
        # L2 Normalize embeddings
        output = F.normalize(output, p=2, dim=1)
        return output

    def get_embedding(self, x):
        return self.forward(x)
