import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


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

        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # Add GeM Pooling and new FC layer
        self.pooling = GeM()
        self.fc = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        """
        Forward pass to generate embeddings.
        """
        # Extract features using DenseNet backbone
        # features shape: (Batch, Channels, H, W)
        features = self.backbone.features(x)

        # DenseNet transition to classifier usually involves ReLU then Pooling
        out = F.relu(features, inplace=True)

        # Apply GeM Pooling
        out = self.pooling(out)

        # Flatten
        out = torch.flatten(out, 1)

        # Embedding Layer
        out = self.fc(out)

        # L2 Normalize embeddings
        output = F.normalize(out, p=2, dim=1)
        return output

    def get_embedding(self, x):
        return self.forward(x)
