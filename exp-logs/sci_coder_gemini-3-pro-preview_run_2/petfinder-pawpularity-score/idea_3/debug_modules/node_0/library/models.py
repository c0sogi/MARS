import torch
import torch.nn as nn
import timm
from library.config import Config


class PawpularityModel(nn.Module):
    """
    Neural network model for Pet Pawpularity Prediction.

    This architecture uses a pre-trained backbone (e.g., Swin Transformer or ConvNeXt)
    to extract image features. These features are then concatenated with 12 binary
    metadata features and passed through a narrow MLP head to predict the target score.

    The model outputs raw logits, suitable for use with BCEWithLogitsLoss.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name (str): Name of the timm backbone to use.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(PawpularityModel, self).__init__()

        # Load the backbone using timm
        # num_classes=0 removes the classification head and returns pooled features
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Determine the output feature dimension of the backbone
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback: infer shape via dummy forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, Config.image_size, Config.image_size)
                dummy_out = self.backbone(dummy_input)
                self.in_features = dummy_out.shape[1]

        # Number of metadata features (Dense features from dataset)
        self.meta_features_dim = 12

        # Define the custom head
        # Structure: Concatenated Input -> Linear -> ReLU -> Dropout -> Linear -> Logits
        self.head = nn.Sequential(
            nn.Linear(self.in_features + self.meta_features_dim, Config.fc_dim),
            nn.ReLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(Config.fc_dim, 1),
        )

    def forward(self, images, metadata):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Batch of images, shape (B, 3, H, W).
            metadata (torch.Tensor): Batch of metadata features, shape (B, 12).

        Returns:
            torch.Tensor: Predicted logits, shape (B, 1).
        """
        # Extract features from the image backbone
        # Shape: (B, in_features)
        img_features = self.backbone(images)

        # Concatenate image features and metadata
        # Shape: (B, in_features + 12)
        combined_features = torch.cat([img_features, metadata], dim=1)

        # Pass through the MLP head
        # Shape: (B, 1)
        logits = self.head(combined_features)

        return logits
