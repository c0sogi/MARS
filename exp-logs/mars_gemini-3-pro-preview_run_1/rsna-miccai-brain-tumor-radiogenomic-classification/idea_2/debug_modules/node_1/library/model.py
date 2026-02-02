import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import BACKBONE, PRETRAINED, HIDDEN_DIM, DROPOUT_RATE


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning.
    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GatedAttention, self).__init__()

        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Bag features of shape (Batch, N, Input_Dim)

        Returns:
            z (torch.Tensor): Aggregated bag representation of shape (Batch, Input_Dim)
            weights (torch.Tensor): Attention weights of shape (Batch, N, 1)
        """
        # V: (Batch, N, Hidden_Dim)
        v = self.attention_V(x)

        # U: (Batch, N, Hidden_Dim)
        u = self.attention_U(x)

        # Gated mechanism: element-wise multiplication
        gated = v * u
        gated = self.dropout(gated)

        # Compute scores: (Batch, N, 1)
        scores = self.attention_weights(gated)

        # Normalize scores to probability distribution over instances (slices)
        weights = F.softmax(scores, dim=1)

        # Weighted sum aggregation
        # x: (Batch, N, Input_Dim) * weights: (Batch, N, 1) -> (Batch, N, Input_Dim)
        # Sum over N -> (Batch, Input_Dim)
        z = torch.sum(x * weights, dim=1)

        return z, weights


class MILNet(nn.Module):
    """
    Multiple Instance Learning Network for MRI Classification.
    Consists of:
    1. Shared Backbone (EfficientNet) for feature extraction per slice.
    2. Gated Attention mechanism for aggregating slice features.
    3. Classifier head for final prediction.
    """

    def __init__(self):
        super(MILNet, self).__init__()

        # 1. Feature Extractor (Instance Level)
        # num_classes=0 returns the pooled feature vector (e.g., 1280 for EffNet-B0)
        self.backbone = timm.create_model(
            BACKBONE, pretrained=PRETRAINED, num_classes=0, in_chans=3
        )

        self.feature_dim = self.backbone.num_features

        # 2. Attention Mechanism (Bag Level)
        self.attention = GatedAttention(
            input_dim=self.feature_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE
        )

        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE), nn.Linear(self.feature_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, N, C, H, W)
                              where N is the number of slices in the bag.
        Returns:
            prob (torch.Tensor): Probability of positive class (Batch, 1)
        """
        batch_size, num_slices, C, H, W = x.shape

        # Collapse Batch and Slice dimensions to process efficiently
        # New shape: (Batch * N, C, H, W)
        x_flat = x.view(batch_size * num_slices, C, H, W)

        # Extract features
        # Output shape: (Batch * N, Feature_Dim)
        features_flat = self.backbone(x_flat)

        # Reshape back to Bag format
        # Shape: (Batch, N, Feature_Dim)
        features = features_flat.view(batch_size, num_slices, -1)

        # Aggregation
        # z shape: (Batch, Feature_Dim)
        z, _ = self.attention(features)

        # Classification
        # prob shape: (Batch, 1)
        prob = self.classifier(z)

        return prob
