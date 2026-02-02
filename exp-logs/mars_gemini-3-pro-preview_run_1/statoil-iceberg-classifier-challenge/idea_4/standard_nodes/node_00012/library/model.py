import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet34(nn.Module):
    """
    ResNet-34 based model for Iceberg classification with Late Fusion of incidence angle.

    Architecture:
    1. Backbone: Pretrained ResNet-34 (ImageNet weights).
    2. Feature Extraction: Convolutional layers + Global Average Pooling -> 512 dim vector.
    3. Late Fusion: Concatenates normalized incidence angle scalar -> 513 dim vector.
    4. Head: BatchNorm -> Dropout -> Dense -> Logits.
    """

    def __init__(self):
        super(IcebergResNet34, self).__init__()

        # Load Pretrained ResNet34 with default (ImageNet) weights
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Capture the number of features from the backbone (512 for ResNet34)
        self.num_features = self.backbone.fc.in_features

        # Remove the original fully connected layer as we will define a custom head
        # We delete it to ensure it's not used or updated, though we won't call it in forward
        del self.backbone.fc

        # Determine the input dimension for the classification head
        self.use_inc_angle = Config.USE_INC_ANGLE

        # Input dimension = Backbone Features (512) + Incidence Angle (1)
        head_input_dim = self.num_features
        if self.use_inc_angle:
            head_input_dim += 1

        # Define the Classification Head
        # As per design: BatchNorm -> Dropout -> Dense
        self.head = nn.Sequential(
            nn.BatchNorm1d(head_input_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(head_input_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, inc_angle=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).
            inc_angle (torch.Tensor, optional): Normalized incidence angles of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # --- Feature Extraction (Backbone) ---
        # We manually pass input through the ResNet layers to skip the deleted fc layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # Shape: (Batch, 512)

        # --- Late Fusion ---
        if self.use_inc_angle:
            if inc_angle is None:
                raise ValueError(
                    "Model configured to use inc_angle, but None provided in forward pass."
                )

            # Ensure inc_angle has shape (Batch, 1) for concatenation
            if inc_angle.dim() == 1:
                inc_angle = inc_angle.unsqueeze(1)

            # Concatenate features with scalar angle
            x = torch.cat((x, inc_angle), dim=1)  # Shape: (Batch, 513)

        # --- Classification Head ---
        logits = self.head(x)

        return logits
