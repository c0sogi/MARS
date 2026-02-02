import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class IcebergResNet18(nn.Module):
    def __init__(self, dropout_rate=0.5):
        """
        Initializes the IcebergResNet18 model.

        Args:
            dropout_rate (float): Probability for the dropout layer in the classification head.
                                  Defaults to 0.5 as per the task specification.
        """
        super(IcebergResNet18, self).__init__()

        # Load ResNet-18 pretrained on ImageNet
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        # Feature Extractor:
        # We keep all layers up to the Global Average Pooling (avgpool).
        # The original ResNet structure ends with:
        #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        #   (fc): Linear(...)
        # list(children)[:-1] removes the 'fc' layer but keeps 'avgpool'.
        # Output shape given 224x224 input: (Batch, 512, 1, 1)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])

        # Minimalist Head for Late Fusion:
        # The input to the head is the concatenated vector of:
        #   - 512 image features (from GAP)
        #   - 1 scalar feature (incidence angle)
        # Total input dimension: 513
        self.bn_fusion = nn.BatchNorm1d(513)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(513, 1)

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, 224, 224).
            angle (torch.Tensor): Normalized incidence angles of shape (Batch,) or (Batch, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Extract Image Features
        # Pass through ResNet backbone
        x = self.features(x)  # Shape: (Batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (Batch, 512)

        # 2. Process Angle
        # Ensure angle tensor has shape (Batch, 1) for concatenation
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # 3. Late Fusion
        # Concatenate image features and angle scalar
        x = torch.cat((x, angle), dim=1)  # Shape: (Batch, 513)

        # 4. Classification Head
        # Apply Batch Normalization to the fused vector
        x = self.bn_fusion(x)
        # Apply Dropout
        x = self.dropout(x)
        # Linear projection to logit
        x = self.fc(x)

        return x
