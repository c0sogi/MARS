import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ArtworkResNet(nn.Module):
    """
    ResNet-34 based model for artwork attribute labeling.
    Replaces the final classification head to match the number of attributes.
    """

    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = Config.PRETRAINED,
        freeze_backbone: bool = Config.FREEZE_BACKBONE,
    ):
        """
        Initializes the ArtworkResNet model.

        Args:
            num_classes (int): Number of output classes (attributes).
            pretrained (bool): Whether to use ImageNet pre-trained weights.
            freeze_backbone (bool): Whether to freeze the backbone weights.
        """
        super(ArtworkResNet, self).__init__()

        # Load ResNet-34 backbone
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = resnet34(weights=weights)

        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        # ResNet-34 has a 'fc' layer at the end. We replace it to output num_classes.
        # The input features for ResNet-34's fc layer is 512.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes).
        """
        # Pass input through the backbone (which includes the modified fc layer)
        logits = self.backbone(x)
        return logits
