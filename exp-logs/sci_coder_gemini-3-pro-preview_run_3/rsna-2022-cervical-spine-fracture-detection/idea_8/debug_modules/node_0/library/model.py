import torch
import torch.nn as nn
import timm
from library.config import Config


class BoxGuidedMILModel(nn.Module):
    """
    Box-Guided 2.5D Contextual MIL Network.

    This model treats the input scan as a sequence of 2.5D slices (3 channels: z-1, z, z+1).
    It extracts features per slice using a ResNet18 backbone, applies a lightweight
    non-linear context module to capture local anatomical continuity, and outputs
    fracture probabilities for each vertebra (C1-C7) at the slice level.

    The aggregation to patient-level predictions (Global Max Pooling) is delegated
    to the loss function during training to enable slice-level supervision via
    bounding boxes.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm backbone (default: resnet18).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super().__init__()

        # 1. Backbone: ResNet18
        # We use num_classes=0 to get the pooled feature vector (e.g., 512 dim for ResNet18)
        # in_chans=3 matches the 2.5D input (z-1, z, z+1)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimension automatically
        self.feature_dim = self.backbone.num_features

        # 2. Context Module: Non-Linear 1D Convolution
        # Structure: Conv1d(k=3, p=1) -> BatchNorm -> ReLU
        # This mixes information across the sequence dimension (S)
        self.context_module = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feature_dim,
                out_channels=self.feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,  # Bias is redundant before BatchNorm
            ),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Instance Classifier
        # Projects context-features to 7 logits (C1-C7) per slice.
        self.classifier = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Channels, Height, Width).
                              Channels should be 3.

        Returns:
            torch.Tensor: Instance logits of shape (Batch, Seq_Len, Num_Classes).
                          Num_Classes is 7 (C1-C7).
        """
        b, s, c, h, w = x.shape

        # 1. TimeDistributed Backbone Application
        # Flatten Batch and Sequence dimensions: (B*S, C, H, W)
        x_flat = x.view(b * s, c, h, w)

        # Extract features: (B*S, Feature_Dim)
        features_flat = self.backbone(x_flat)

        # Reshape to recover sequence dimension: (B, S, Feature_Dim)
        # Then permute to (B, Feature_Dim, S) for Conv1d
        features_seq = features_flat.view(b, s, self.feature_dim).permute(0, 2, 1)

        # 2. Apply Context Module
        # Input: (B, Feature_Dim, S) -> Output: (B, Feature_Dim, S)
        context_features = self.context_module(features_seq)

        # Permute back to (B, S, Feature_Dim) for Linear layer
        context_features = context_features.permute(0, 2, 1)

        # 3. Instance Classifier
        # Input: (B, S, Feature_Dim) -> Output: (B, S, Num_Classes)
        instance_logits = self.classifier(context_features)

        return instance_logits
