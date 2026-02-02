import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based architecture for Ship vs Iceberg classification.

    This model implements the 'Calibrated Cosine-SWA ResNet-18 Ensemble' architecture:
    1. Backbone: Pretrained ResNet-18 (ImageNet weights).
    2. Pooling: Global Average Pooling (GAP).
    3. Fusion: Late fusion of the scalar 'inc_angle' with the GAP feature vector.
    4. Head: Minimalist head (BatchNorm -> Dropout -> Linear).
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet-18
        # We attempt to use the modern weights API, falling back to the legacy method if needed.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.resnet = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.resnet = models.resnet18(pretrained=True)

        # The output feature dimension of the ResNet-18 backbone (before FC) is 512.
        self.num_features = self.resnet.fc.in_features

        # Replace the original FC layer with Identity.
        # We will bypass this in forward(), but this ensures no unused parameters are tracked
        # if one were to call resnet(x) directly.
        self.resnet.fc = nn.Identity()

        # Minimalist Classification Head
        # Input: 512 (Image Features) + 1 (Incidence Angle) = 513
        # Structure: Batch Normalization -> Dropout -> Linear Output
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features + 1),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.num_features + 1, 1),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (Batch,) or (Batch, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # --- Backbone Feature Extraction ---
        # Manually pass through ResNet layers to extract features before the original FC layer.

        # Stem layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Residual blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Global Average Pooling
        x = self.resnet.avgpool(x)

        # Flatten the output: (Batch, 512, 1, 1) -> (Batch, 512)
        x = torch.flatten(x, 1)

        # --- Late Fusion ---
        # Ensure angle has the correct shape for concatenation: (Batch, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Concatenate the image features with the incidence angle
        # x: (Batch, 512), angle: (Batch, 1) -> combined: (Batch, 513)
        x = torch.cat((x, angle), dim=1)

        # --- Classification Head ---
        # Pass fused vector through the minimalist head to get logits
        logits = self.head(x)

        return logits
