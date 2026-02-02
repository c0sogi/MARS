import torch
import torch.nn as nn
import timm
from library.config import Config


class PawpularitySwinModel(nn.Module):
    """
    Swin Transformer based model for Pawpularity prediction.
    Fuses image features from the backbone with tabular metadata features.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the model architecture.

        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights for the backbone.
        """
        super(PawpularitySwinModel, self).__init__()

        # 1. Backbone: Swin Transformer Tiny
        # We set num_classes=0 to remove the classification head and return the
        # global pooled feature vector (embedding).
        self.backbone = timm.create_model(
            Config.model_name, pretrained=pretrained, num_classes=0
        )

        # Get the dimension of the backbone output features (e.g., 768 for Swin Tiny)
        self.n_backbone_features = self.backbone.num_features

        # 2. Metadata Features
        # Calculate the number of additional binary features from the config
        self.n_meta_features = len(Config.feature_cols)

        # 3. Fusion & Prediction Head (MLP)
        # The input dimension for the head is the sum of the image embedding dimension
        # and the number of metadata features.
        self.input_dim = self.n_backbone_features + self.n_meta_features

        # MLP Structure: Projection -> BatchNorm -> Activation -> Dropout -> Prediction
        # This introduces non-linearity to model interactions between visual and metadata features.
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, Config.fc_dim),
            nn.BatchNorm1d(Config.fc_dim),
            nn.SiLU(),  # Swish activation (often preferred for Transformers/ConvNeXt)
            nn.Dropout(Config.dropout),
            nn.Linear(Config.fc_dim, 1),  # Output is a single scalar (logit)
        )

    def forward(self, images, features):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Batch of images, shape (Batch_Size, Channels, Height, Width)
            features (torch.Tensor): Batch of metadata features, shape (Batch_Size, N_Meta_Features)

        Returns:
            torch.Tensor: Logits for Pawpularity score, shape (Batch_Size, 1)
        """
        # Extract image features using the backbone
        # Shape: (Batch_Size, n_backbone_features)
        img_features = self.backbone(images)

        # Ensure metadata features are on the correct device and dtype to match image features
        features = features.to(img_features.device).type(img_features.dtype)

        # Concatenate image features and metadata features along the feature dimension
        # Shape: (Batch_Size, n_backbone_features + n_meta_features)
        combined_features = torch.cat([img_features, features], dim=1)

        # Pass the fused vector through the MLP head to get the prediction
        # Shape: (Batch_Size, 1)
        output = self.mlp(combined_features)

        return output
