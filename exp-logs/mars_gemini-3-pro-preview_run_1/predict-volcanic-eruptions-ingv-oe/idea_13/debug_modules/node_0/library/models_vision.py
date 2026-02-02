import torch
import torch.nn as nn
import timm
from library.config import Config


class ScalarConditionedEfficientNet(nn.Module):
    """
    A Hybrid Vision-Scalar Architecture for Volcano Eruption Prediction.

    This model combines:
    1. An EfficientNet-B0 backbone to extract texture and pattern features from
       Log-Mel Spectrograms (10 channels).
    2. A Scalar Injection path to provide absolute energy statistics (30 features)
       directly to the regression head, bypassing internal normalization layers.

    This design addresses 'Magnitude-Dependent Residuals' by ensuring the model
    can distinguish between spectrally similar but energetically distinct signals.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load ImageNet-1k pretrained weights for the backbone.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Vision Backbone (EfficientNet-B0)
        # ------------------------------------------------------------------
        # We use timm to create the model.
        # - in_chans=Config.IN_CHANNELS (10): Adapts the first conv layer for 10 sensors.
        # - num_classes=0: Removes the default classification head.
        # - global_pool='avg': Applies Global Average Pooling to output a 1D vector.
        self.backbone = timm.create_model(
            Config.CNN_MODEL_NAME,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # Retrieve the output feature dimension of the backbone
        # For EfficientNet-B0, this is typically 1280.
        self.backbone_dim = self.backbone.num_features

        # ------------------------------------------------------------------
        # 2. Scalar Injection & Regression Head
        # ------------------------------------------------------------------
        self.scalar_dim = Config.SCALAR_INPUT_DIM

        # The head receives the concatenation of image features and scalar features
        combined_dim = self.backbone_dim + self.scalar_dim

        # MLP Regression Head
        # Structure: Dropout -> Linear -> SiLU -> Dropout -> Linear -> Output
        # SiLU (Swish) is used to maintain consistency with EfficientNet's activations.
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(combined_dim, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x_img, x_scalar):
        """
        Forward pass of the model.

        Args:
            x_img (torch.Tensor): Spectrogram input of shape (Batch, 10, H, W).
            x_scalar (torch.Tensor): Scalar statistics input of shape (Batch, 30).

        Returns:
            torch.Tensor: Predicted time_to_eruption (Log-Transformed scale) of shape (Batch, 1).
        """
        # 1. Extract features from Spectrograms
        # Output shape: (Batch, backbone_dim)
        img_features = self.backbone(x_img)

        # 2. Prepare Scalars
        # Ensure correct dtype (float32) matches the network weights
        x_scalar = x_scalar.type_as(img_features)

        # 3. Feature Fusion (Concatenation)
        # Combine the learned spectral representation with absolute energy stats
        # Output shape: (Batch, backbone_dim + scalar_dim)
        combined_features = torch.cat([img_features, x_scalar], dim=1)

        # 4. Regression
        # Output shape: (Batch, 1)
        output = self.head(combined_features)

        return output
