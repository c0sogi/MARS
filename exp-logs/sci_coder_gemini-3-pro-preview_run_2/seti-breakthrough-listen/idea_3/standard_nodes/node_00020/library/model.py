import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the p-norm of the input tensor.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, L) where L is the temporal dimension
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Pools over the last dimension (L)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)


class SETIModel(nn.Module):
    """
    Hybrid ConvNeXt-1D model for SETI signal detection.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (2D CNN) to extract spatial features.
    2. Frequency Pooling: Collapses the frequency dimension (Width) to retain temporal structure.
    3. 1D Conv Head: Processes the temporal sequence to detect cadence patterns.
    4. GeM Pooling: Aggregates the temporal dimension to isolate sparse signals.
    5. Classifier: Linear layer for binary prediction.
    """

    def __init__(self, pretrained=True):
        super(SETIModel, self).__init__()

        # Initialize Backbone
        # features_only=True returns a list of feature maps.
        # out_indices=(3,) selects the output of the final stage.
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=pretrained,
            in_chans=Config.in_channels,
            features_only=True,
            out_indices=(3,),
        )

        # Dynamically determine the number of input features from the backbone
        # We run a dummy forward pass to inspect the shape.
        dummy_input = torch.zeros(1, Config.in_channels, 256, 256)
        with torch.no_grad():
            features_list = self.backbone(dummy_input)
            last_feat = features_list[-1]
            in_features = last_feat.shape[1]

        # 1D Convolutional Block
        # Processes the sequence (B, C, H) where H is the time axis.
        self.conv1d_head = nn.Sequential(
            nn.Conv1d(in_features, in_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
        )

        # Generalized Mean Pooling
        self.gem = GeM(p=3.0)

        # Classification Head
        self.drop = nn.Dropout(Config.drop_rate)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, 1638, 256)
        Returns:
            logits: Output tensor of shape (B, 1)
        """
        # 1. Backbone Feature Extraction
        # Returns a list of feature maps; we take the last one.
        # Shape: (B, C, H_feat, W_feat)
        # H_feat corresponds to the Time axis, W_feat to the Frequency axis.
        features_list = self.backbone(x)
        x = features_list[-1]

        # 2. Frequency Pooling
        # Global Average Pooling along the Frequency (Width) axis.
        # Collapses (B, C, H, W) -> (B, C, H)
        x = x.mean(dim=3)

        # 3. Cadence-Aware 1D Processing
        # Input: (B, C, H_feat) representing the temporal sequence.
        x = self.conv1d_head(x)

        # 4. Temporal Pooling (GeM)
        # Aggregates the temporal dimension to a single vector.
        # (B, C, H) -> (B, C, 1) -> (B, C)
        x = self.gem(x)
        x = x.squeeze(-1)

        # 5. Classifier
        x = self.drop(x)
        logits = self.fc(x)

        return logits
