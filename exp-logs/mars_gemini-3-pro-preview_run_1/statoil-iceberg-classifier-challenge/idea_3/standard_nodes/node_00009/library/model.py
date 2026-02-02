import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, DROPOUT_RATE, NUM_CLASSES


class IcebergModel(nn.Module):
    """
    CNN based model for Ship vs Iceberg classification.
    Uses Late Fusion to incorporate incidence angle information.
    """

    def __init__(self):
        super(IcebergModel, self).__init__()

        # Load pretrained backbone
        # num_classes=0 removes the classifier head and pooling, returning the features
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)

        # Get the number of features output by the backbone
        num_features = self.backbone.num_features

        # Define the classification head
        # Input dimension = Image Features + 1 (Incidence Angle)
        self.head_input_dim = num_features + 1

        # Batch Normalization on the fused vector
        self.bn = nn.BatchNorm1d(self.head_input_dim)

        # Dropout
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

        # Final Linear Layer
        self.fc = nn.Linear(self.head_input_dim, NUM_CLASSES)

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).
            angle (torch.Tensor): Incidence angles of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Extract features from images
        # Shape: (Batch, num_features)
        features = self.backbone(x)

        # Ensure angle is (Batch, 1)
        angle = angle.view(-1, 1)

        # Late Fusion: Concatenate features and angle
        # Shape: (Batch, num_features + 1)
        combined = torch.cat((features, angle), dim=1)

        # Apply Head Layers
        x = self.bn(combined)
        x = self.dropout(x)
        x = self.fc(x)

        return x
