import torch
import torch.nn as nn
from torchvision import models
from library import config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg detection with Late Fusion of incidence angle.

    Architecture:
    1. Backbone: ResNet-18 (pretrained)
    2. Pooling: Global Average Pooling -> 512 dim
    3. Fusion: Concatenate 512 dim features + 1 dim incidence angle -> 513 dim
    4. Head: BatchNorm -> Dropout -> Linear -> Logits
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet18
        # Using the modern 'weights' parameter if available, otherwise fallback to pretrained=True
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=True)

        # We will use the backbone up to the avgpool layer.
        # Standard ResNet structure:
        # conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc

        # Remove the fully connected layer as we will define our own head
        del self.backbone.fc

        # Define the classification head
        # Input: 512 (image features) + 1 (angle) = 513
        self.head = nn.Sequential(
            nn.BatchNorm1d(config.FEAT_DIM + 1),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(config.FEAT_DIM + 1, config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W)
            angle (torch.Tensor): Incidence angle tensor of shape (B,) or (B, 1)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # --- Backbone Feature Extraction ---
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        # Flatten: (B, 512, 1, 1) -> (B, 512)
        x = torch.flatten(x, 1)

        # --- Late Fusion ---
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate features and angle
        x = torch.cat([x, angle], dim=1)

        # --- Classification Head ---
        logits = self.head(x)

        return logits
