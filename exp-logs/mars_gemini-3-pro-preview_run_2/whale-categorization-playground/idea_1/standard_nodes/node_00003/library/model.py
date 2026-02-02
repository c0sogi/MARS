import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class EmbeddingNet(nn.Module):
    """
    A neural network that maps images to an embedding space.
    Uses ResNet-18 as the backbone.
    """

    def __init__(self):
        super(EmbeddingNet, self).__init__()

        # Determine weights based on configuration
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        # Load the backbone
        self.backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet18's fc layer has 512 input features
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.EMBEDDING_DIM)

    def forward(self, x):
        """
        Forward pass of the embedding network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: L2-normalized embeddings of shape (B, EMBEDDING_DIM).
        """
        x = self.backbone(x)
        # L2-normalize the output vectors
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNet(nn.Module):
    """
    Siamese Network wrapper that processes pairs of images using a shared EmbeddingNet.
    """

    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        """
        Forward pass for image pairs.

        Args:
            x1 (torch.Tensor): First batch of images.
            x2 (torch.Tensor): Second batch of images.

        Returns:
            tuple: (output1, output2) embeddings.
        """
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Minimizes distance for positive pairs (label=1) and maximizes distance
    for negative pairs (label=0) up to a margin.
    """

    def __init__(self, margin=Config.MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Args:
            output1 (torch.Tensor): Embeddings for first images.
            output2 (torch.Tensor): Embeddings for second images.
            label (torch.Tensor): 1.0 for same class, 0.0 for different class.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Formula: Y * D^2 + (1 - Y) * max(0, margin - D)^2
        # Note: dataset.py yields label=1 for same, label=0 for different.

        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
