import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a pretrained CNN backbone.

    This model uses a backbone from the `timm` library (e.g., ResNet-18) to extract
    features from spectrogram images. The original classification head is replaced
    with a custom fully connected layer that outputs raw logits for 19 bird species.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
        in_channels: int = Config.IN_CHANNELS,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        """
        Args:
            model_name (str): Name of the backbone model (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of target classes (bird species).
            in_channels (int): Number of input channels (3 for RGB).
            dropout_rate (float): Dropout probability before the final layer.
        """
        super(BirdClassifier, self).__init__()

        # Initialize the backbone using timm
        # num_classes=0 and global_pool='avg' removes the default head and
        # returns the pooled feature vector.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
        )

        # Retrieve the feature dimension size from the backbone
        in_features = self.backbone.num_features

        # Define the custom classification head
        # We use a Sequential container to apply Dropout followed by the Linear projection.
        # The output is raw logits, compatible with BCEWithLogitsLoss.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.head(features)

        return logits
