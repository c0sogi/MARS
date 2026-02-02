import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MILNet(nn.Module):
    """
    Attention-Gated Multiple Instance Learning (MIL) Network.

    Architecture:
    1. Feature Extractor: EfficientNet-B0 (pretrained, 4 input channels).
       Extracts a feature vector from each slice independently.
    2. Gated Attention Module: Determines the importance (weight) of each slice.
    3. Aggregation: Computes weighted sum of slice features.
    4. Classifier: Predicts MGMT_value from the aggregated feature.
    """

    def __init__(self, in_channels=4, hidden_dim=512):
        super(MILNet, self).__init__()

        # 1. Feature Extractor
        # Use timm to create EfficientNet-B0.
        # in_chans=4 adapts the first conv layer.
        # num_classes=0 returns the pooled feature vector (1280 dim for B0).
        self.feature_extractor = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=in_channels, num_classes=0
        )

        # EfficientNet-B0 output feature dimension
        self.feature_dim = self.feature_extractor.num_features

        # 2. Gated Attention Mechanism
        # Formula: a = w^T (tanh(V h) * sigmoid(U h))
        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hidden_dim, 1)

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Num_Slices, Channels, H, W).

        Returns:
            logits (torch.Tensor): Output logits of shape (Batch_Size, 1).
        """
        batch_size, num_slices, C, H, W = x.shape

        # Collapse Batch and Slice dimensions to process all images through CNN in parallel
        # Shape: (Batch_Size * Num_Slices, C, H, W)
        x_reshaped = x.view(batch_size * num_slices, C, H, W)

        # Extract features
        # Shape: (Batch_Size * Num_Slices, Feature_Dim)
        features = self.feature_extractor(x_reshaped)

        # Restore Batch and Slice dimensions
        # Shape: (Batch_Size, Num_Slices, Feature_Dim)
        features = features.view(batch_size, num_slices, -1)

        # --- Attention Mechanism ---

        # V: (Batch_Size, Num_Slices, Hidden_Dim)
        A_V = self.attention_V(features)

        # U: (Batch_Size, Num_Slices, Hidden_Dim)
        A_U = self.attention_U(features)

        # Gated combination
        # Shape: (Batch_Size, Num_Slices, Hidden_Dim)
        gated_attention = A_V * A_U

        # Compute raw attention scores
        # Shape: (Batch_Size, Num_Slices, 1)
        A_scores = self.attention_weights(gated_attention)

        # Normalize scores using Softmax over the slice dimension (dim=1)
        # Shape: (Batch_Size, Num_Slices, 1)
        A_weights = F.softmax(A_scores, dim=1)

        # --- Aggregation ---

        # Weighted sum of features
        # Transpose A_weights to (Batch, 1, Slices) and bmm with features (Batch, Slices, Feat)
        # Result: (Batch, 1, Feat) -> Squeeze to (Batch, Feat)
        # Alternatively: sum(features * weights, dim=1)

        # (Batch, Slices, Feat) * (Batch, Slices, 1) -> (Batch, Slices, Feat)
        weighted_features = features * A_weights

        # Sum over slices
        # Shape: (Batch, Feature_Dim)
        patient_embedding = torch.sum(weighted_features, dim=1)

        # --- Classifier ---

        # Shape: (Batch, 1)
        logits = self.classifier(patient_embedding)

        return logits
