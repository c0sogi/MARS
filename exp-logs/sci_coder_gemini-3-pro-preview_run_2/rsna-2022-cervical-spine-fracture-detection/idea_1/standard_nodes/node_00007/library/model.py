import torch
import torch.nn as nn
import timm
from library.config import Config


class FractureMILModel(nn.Module):
    """
    2D Multiple Instance Learning (MIL) Model for Cervical Spine Fracture Detection.

    This model treats a 3D CT scan as a 'bag' of 2D slices.
    It uses Embedding-Space Aggregation (Attention) to combine slice features
    into a study-level representation before classification.
    Cite solution_lesson_node_00005
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        """
        Args:
            model_name (str): The name of the timm backbone to use (default: efficientnet_b0).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(FractureMILModel, self).__init__()

        # Initialize the 2D CNN backbone using timm.
        # num_classes=0 returns the global pool features (embedding) instead of logits.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
        )

        self.feature_dim = self.backbone.num_features

        # Attention Mechanism
        # Learn to weight slices based on their feature content
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        # Final Classifier
        # Maps aggregated study embedding to class probabilities
        self.classifier = nn.Linear(self.feature_dim, Config.N_CLASSES)

    def forward(self, x):
        """
        Forward pass of the MIL model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width).

        Returns:
            torch.Tensor: Predicted probabilities of shape (Batch, N_CLASSES).
        """
        # Unpack dimensions
        b, s, c, h, w = x.shape

        # 1. Collapse Batch and Slice dimensions
        x = x.view(b * s, c, h, w)

        # 2. Slice-level Feature Extraction
        # Shape: (Batch * Slices, FeatureDim)
        features = self.backbone(x)

        # 3. Reshape back to Study structure
        # Shape: (Batch, Slices, FeatureDim)
        features = features.view(b, s, self.feature_dim)

        # 4. Attention Aggregation
        # Calculate attention scores
        attn_weights = self.attention(features)  # (Batch, Slices, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of features
        # Shape: (Batch, FeatureDim)
        study_embedding = torch.sum(features * attn_weights, dim=1)

        # 5. Classification
        logits = self.classifier(study_embedding)
        probs = torch.sigmoid(logits)

        return probs
