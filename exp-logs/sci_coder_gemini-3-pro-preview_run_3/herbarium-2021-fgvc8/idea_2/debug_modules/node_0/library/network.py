import torch
import torch.nn as nn
import timm
from library.config import Config


class PlantClassifier(nn.Module):
    """
    PlantClassifier model architecture.

    Uses a ConvNeXt-Small backbone to extract 768-dimensional feature embeddings,
    followed by a monolithic linear classification head mapping to 64,500 classes.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pretrained ImageNet weights for the backbone.
        """
        super(PlantClassifier, self).__init__()

        # Initialize the backbone using timm
        # num_classes=0 ensures we get the feature vector (pooled and normalized)
        # rather than the default classifier output.
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # Determine input features for the classification head
        # ConvNeXt-Small typically has 768 features
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            in_features = Config.EMBEDDING_DIM

        # Monolithic Linear Classification Head
        # Maps (Batch, 768) -> (Batch, 64500)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Output shape: (Batch, 768)
        features = self.backbone(x)

        # Compute class logits
        # Output shape: (Batch, 64500)
        logits = self.fc(features)

        return logits
