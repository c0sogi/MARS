import torch
import torch.nn as nn
import timm
from library.config import Config


class AdaptiveBackbone(nn.Module):
    """
    Stage 1 Model: Adaptive Feature Backbones.

    This architecture combines two powerful pre-trained backbones:
    1. Swin Transformer (Global composition)
    2. EfficientNetV2 (Local detail)

    The outputs of these backbones are concatenated to form a rich embedding space.
    During the fine-tuning phase (Stage 1), a temporary regression head is used
    to adapt the feature manifold to the 'Pawpularity' task.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the dual-backbone architecture.

        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
                               Defaults to True.
        """
        super(AdaptiveBackbone, self).__init__()

        # Initialize Swin Transformer Backbone
        # num_classes=0 removes the classification head and applies global pooling,
        # returning a feature vector of shape (B, num_features).
        self.swin = timm.create_model(
            Config.BACKBONE_SWIN, pretrained=pretrained, num_classes=0
        )

        # Initialize EfficientNetV2 Backbone
        self.effnet = timm.create_model(
            Config.BACKBONE_EFFNET, pretrained=pretrained, num_classes=0
        )

        # Determine the dimension of the concatenated features
        self.swin_dim = self.swin.num_features
        self.effnet_dim = self.effnet.num_features
        self.embedding_dim = self.swin_dim + self.effnet_dim

        # Temporary Regression Head
        # Consists of a Linear layer mapping the embedding to a single scalar,
        # followed by a Sigmoid activation to constrain output to [0, 1].
        # This head is used for training the backbones but discarded for Stage 2.
        self.head = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, feature_extract: bool = False) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch_Size, Channels, Height, Width).
            feature_extract (bool): If True, returns the concatenated feature embeddings
                                    instead of the regression score. Used for Stage 2.

        Returns:
            torch.Tensor:
                - If feature_extract=False: Regression score (B, 1) in range [0, 1].
                - If feature_extract=True: Concatenated embeddings (B, embedding_dim).
        """
        # 1. Extract features from Swin Transformer
        # Shape: (B, swin_dim)
        f_swin = self.swin(x)

        # 2. Extract features from EfficientNetV2
        # Shape: (B, effnet_dim)
        f_effnet = self.effnet(x)

        # 3. Concatenate features
        # Shape: (B, swin_dim + effnet_dim)
        embeddings = torch.cat([f_swin, f_effnet], dim=1)

        # 4. Return embeddings or prediction based on mode
        if feature_extract:
            return embeddings

        # 5. Compute regression score
        # Shape: (B, 1)
        out = self.head(embeddings)
        return out
