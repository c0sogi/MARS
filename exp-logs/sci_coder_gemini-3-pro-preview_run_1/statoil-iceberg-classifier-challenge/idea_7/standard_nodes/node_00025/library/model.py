import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg detection with Late Fusion of Incidence Angle.

    Architecture:
    1. Backbone: Pretrained ResNet-18 (ImageNet weights).
    2. Pooling: Global Average Pooling (inherent in ResNet).
    3. Fusion: Concatenation of 512-dim image features with 1-dim incidence angle.
    4. Head: BatchNorm -> Dropout -> Dense -> Output.
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet-18 backbone
        # We use the default weights (ImageNet) which are optimal for transfer learning
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Get the number of input features for the final FC layer (512 for ResNet-18)
        num_ftrs = self.backbone.fc.in_features

        # Replace the final fully connected layer with Identity.
        # The torchvision ResNet implementation performs Global Average Pooling
        # followed by flattening before the FC layer.
        # By setting FC to Identity, we extract the (Batch, 512) feature vector.
        self.backbone.fc = nn.Identity()

        # Define the classification head with Late Fusion
        # Input: 512 image features + 1 incidence angle feature
        fusion_input_dim = num_ftrs + 1

        self.head = nn.Sequential(
            nn.BatchNorm1d(fusion_input_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(fusion_input_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, inc_angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).
            inc_angle (torch.Tensor): Incidence angles of shape (Batch,) or (Batch, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Pass images through the backbone to get features
        # Shape: (Batch, 512)
        img_features = self.backbone(x)

        # Ensure incidence angle has the correct shape for concatenation
        # Shape: (Batch, 1)
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.view(-1, 1)

        # Late Fusion: Concatenate image features with incidence angle
        # Shape: (Batch, 513)
        combined_features = torch.cat((img_features, inc_angle), dim=1)

        # Pass through the head to get logits
        # Shape: (Batch, 1)
        logits = self.head(combined_features)

        return logits
