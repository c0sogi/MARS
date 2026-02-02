import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class HotelClassifier(nn.Module):
    """
    Hotel ID Classifier using a ResNet-18 backbone.

    The architecture consists of a pre-trained ResNet-18 model where the
    final fully connected layer is replaced to map the 512-dimensional
    feature vector to the specific number of hotel classes.
    """

    def __init__(
        self,
        n_classes: int = Config.NUM_CLASSES,
        pretrained: bool = Config.PRETRAINED,
        dropout: float = Config.DROPOUT,
    ):
        """
        Initialize the HotelClassifier.

        Args:
            n_classes (int): Number of target classes (hotels). Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to use ImageNet pre-trained weights. Defaults to Config.PRETRAINED.
            dropout (float): Dropout probability for the final layer. Defaults to Config.DROPOUT.
        """
        super(HotelClassifier, self).__init__()

        # Select weights based on pretrained flag
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-18 backbone
        self.backbone = resnet18(weights=weights)

        # Get the input dimension of the original fully connected layer (usually 512 for ResNet18)
        in_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        # We include Dropout for regularization as specified in the configuration strategy
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_features, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits for each class. Shape (B, n_classes).
        """
        return self.backbone(x)
