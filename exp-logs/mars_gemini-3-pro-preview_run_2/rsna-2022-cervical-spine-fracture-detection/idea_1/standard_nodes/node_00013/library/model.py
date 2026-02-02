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
        # Cite solution_lesson_node_00005: Use Embedding-Space Aggregation.
        # num_classes=0 returns the pooled feature vector (B, num_features).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
        )
        self.in_features = self.backbone.num_features

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.in_features, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT), nn.Linear(self.in_features, Config.N_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass of the MIL model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width).

        Returns:
            torch.Tensor: Predicted probabilities of shape (Batch, N_CLASSES).
        """
        b, s, c, h, w = x.shape

        # 1. Collapse Batch and Slice dimensions
        x = x.view(b * s, c, h, w)

        # 2. Feature Extraction
        # Shape: (Batch * Slices, Features)
        features = self.backbone(x)

        # 3. Reshape back to Study structure
        # Shape: (Batch, Slices, Features)
        features = features.view(b, s, self.in_features)

        # 4. Attention Aggregation (Cite solution_lesson_node_00005)
        # Calculate attention weights: (Batch, Slices, 1)
        attn_weights = self.attention(features)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum of features: (Batch, Features)
        study_embedding = torch.sum(features * attn_weights, dim=1)

        # 5. Classification
        logits = self.classifier(study_embedding)

        return torch.sigmoid(logits)
