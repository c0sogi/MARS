import torch
import torch.nn as nn
import timm
from library.config import Config


class FractureMILModel(nn.Module):
    """
    2D Multiple Instance Learning (MIL) Model for Cervical Spine Fracture Detection.

    This model treats a 3D CT scan as a 'bag' of 2D slices. It processes each slice
    independently using a 2D CNN backbone and aggregates the predictions using
    Max-Pooling to generate a study-level probability for each fracture type.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        """
        Args:
            model_name (str): The name of the timm backbone to use (default: efficientnet_b0).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(FractureMILModel, self).__init__()

        # Initialize the 2D CNN backbone using timm.
        # - in_chans=1: Adapts the first layer for grayscale CT inputs.
        # - num_classes=0: Returns the global pooled feature vector instead of logits.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
        )

        self.feature_dim = self.backbone.num_features

        # Attention Mechanism (Cite solution_lesson_node_00002)
        # Learns to weight slices dynamically instead of using Max-Pooling
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(self.feature_dim, Config.N_CLASSES),
        )

    def forward(self, x):
        """
        Forward pass of the MIL model with Attention Pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width).

        Returns:
            torch.Tensor: Predicted probabilities of shape (Batch, N_CLASSES).
        """
        # Unpack dimensions
        b, s, c, h, w = x.shape

        # 1. Collapse Batch and Slice dimensions
        x = x.view(b * s, c, h, w)

        # 2. Extract Features
        # Shape: (Batch * Slices, Feature_Dim)
        features = self.backbone(x)

        # 3. Reshape back to Study structure
        # Shape: (Batch, Slices, Feature_Dim)
        features = features.view(b, s, -1)

        # 4. Calculate Attention Weights
        # Shape: (Batch, Slices, 1)
        attn_weights = self.attention(features)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # 5. Weighted Aggregation
        # Shape: (Batch, Feature_Dim)
        study_embedding = torch.sum(features * attn_weights, dim=1)

        # 6. Classification
        # Shape: (Batch, N_CLASSES)
        logits = self.classifier(study_embedding)

        # 7. Activation
        probs = torch.sigmoid(logits)

        return probs
