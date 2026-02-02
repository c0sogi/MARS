import torch
import torch.nn as nn
import timm
from library.config import Config


class S3DNet(nn.Module):
    """
    Siamese Spatially-Strided 2.5D Network (S3D-Net).

    This model processes high-density volumetric MRI data by splitting it into two
    interleaved streams (Even and Odd slices). Both streams are processed by a
    shared 2.5D backbone (EfficientNet-B0) to extract features at native resolution
    without exploding channel depth. The features are fused via concatenation and
    passed to a classification head.
    """

    def __init__(self):
        super(S3DNet, self).__init__()

        # ----------------------------------------------------------------------
        # Shared Backbone
        # ----------------------------------------------------------------------
        # We use EfficientNet-B0.
        # - in_chans=64: Adapts the first layer to accept (16 slices * 4 modalities).
        # - num_classes=0: Removes the default FC layer and returns the pooled feature vector.
        # - drop_path_rate: Enables Stochastic Depth for regularization.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            in_chans=Config.IN_CHANS,
            num_classes=0,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Retrieve the feature dimension of the backbone (e.g., 1280 for EfficientNet-B0)
        self.num_features = self.backbone.num_features

        # ----------------------------------------------------------------------
        # Fusion Head
        # ----------------------------------------------------------------------
        # Late Fusion: We concatenate the feature vectors from the Even and Odd streams.
        # Input dimension becomes num_features * 2.
        # Output dimension is 1 (logit for binary classification).
        self.classifier = nn.Linear(self.num_features * 2, 1)

    def forward(self, x_even, x_odd):
        """
        Forward pass of the Siamese Network.

        Args:
            x_even (torch.Tensor): Input tensor for the Even stream.
                                   Shape: (Batch_Size, 64, 224, 224)
            x_odd (torch.Tensor):  Input tensor for the Odd stream.
                                   Shape: (Batch_Size, 64, 224, 224)

        Returns:
            torch.Tensor: Logits representing the probability of MGMT promoter methylation.
                          Shape: (Batch_Size, 1)
        """
        # 1. Shared Feature Extraction
        # Pass both streams through the same backbone instance (weight sharing).
        # The backbone performs the forward pass and Global Average Pooling.
        feat_even = self.backbone(x_even)  # Shape: (B, num_features)
        feat_odd = self.backbone(x_odd)  # Shape: (B, num_features)

        # 2. Late Fusion
        # Concatenate the feature vectors along the channel dimension.
        combined_features = torch.cat(
            [feat_even, feat_odd], dim=1
        )  # Shape: (B, num_features * 2)

        # 3. Classification
        logits = self.classifier(combined_features)  # Shape: (B, 1)

        return logits
