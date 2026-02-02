import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multi-Instance Learning.
    References: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    """

    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features (Batch, Views, Feature_Dim)
            mask: Binary mask (Batch, Views), 1 for valid, 0 for padding.
        Returns:
            weighted_features: (Batch, Feature_Dim)
            attention_scores: (Batch, Views, 1)
        """
        # Calculate gated attention scores
        # V(x) -> Tanh, U(x) -> Sigmoid
        # Gated = V(x) * U(x)
        a_v = self.attention_V(x)  # (B, V, Attn_Dim)
        a_u = self.attention_U(x)  # (B, V, Attn_Dim)

        # Project to scalar score
        a = self.attention_weights(a_v * a_u)  # (B, V, 1)

        # Apply masking if provided
        if mask is not None:
            # Expand mask to match attention score dimensions
            mask_expanded = mask.unsqueeze(-1)  # (B, V, 1)
            # Set scores of padded instances to a very large negative number
            # so that softmax results in 0 probability.
            a = a.masked_fill(mask_expanded == 0, -1e9)

        # Normalize scores via Softmax over the views dimension
        attn_scores = F.softmax(a, dim=1)  # (B, V, 1)

        # Compute weighted sum of instance features
        # (B, V, 1) * (B, V, D) -> (B, V, D) -> sum -> (B, D)
        weighted_features = torch.sum(x * attn_scores, dim=1)

        return weighted_features, attn_scores


class BreastCancerMILModel(nn.Module):
    """
    End-to-End Multi-View Attention Network (MVAN).

    Architecture:
    1. Backbone: EfficientNet-B2 (Fine-tuned) extracts features from each view.
    2. Aggregation: Gated Attention pools view features into a breast embedding.
    3. Fusion: Metadata (Age, Implant, Machine) is processed and concatenated.
    4. Head: Binary classification.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Image Encoder Backbone
        # We use global_pool='avg' to get a 1D feature vector per image.
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,  # Remove classification head
            global_pool="avg",
            drop_rate=config.DROPOUT,
        )

        # Determine backbone output dimension dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            self.feature_dim = self.backbone(dummy_input).shape[1]

        # 2. MIL Aggregation Layer
        self.attention = GatedAttention(
            input_dim=self.feature_dim, attention_dim=config.ATTENTION_DIM
        )

        # 3. Metadata Processing Branch
        # Input: Age (1), Implant (1), MachineID (1) -> Total 3
        self.meta_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # 4. Final Classification Head
        # Concatenates Image Embedding (feature_dim) + Metadata Embedding (32)
        classifier_input_dim = self.feature_dim + 32
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, images, mask, metadata):
        """
        Args:
            images: Tensor of shape (Batch, Max_Views, Channels, Height, Width)
            mask: Tensor of shape (Batch, Max_Views) indicating valid images.
            metadata: Tensor of shape (Batch, 3) containing [age, implant, machine_id].

        Returns:
            logits: (Batch, 1)
        """
        b, v, c, h, w = images.shape

        # --- Step 1: Feature Extraction ---
        # Flatten batch and views dimensions to process all images in parallel
        # Shape: (B * V, C, H, W)
        x = images.view(b * v, c, h, w)

        # Pass through backbone
        # Shape: (B * V, Feature_Dim)
        features = self.backbone(x)

        # Reshape back to bag structure
        # Shape: (B, V, Feature_Dim)
        features = features.view(b, v, -1)

        # --- Step 2: Aggregation ---
        # Apply Gated Attention to pool features
        # bag_embedding: (B, Feature_Dim)
        bag_embedding, _ = self.attention(features, mask)

        # --- Step 3: Metadata Fusion ---
        # Process metadata
        # meta_embedding: (B, 32)
        meta_embedding = self.meta_mlp(metadata)

        # Concatenate visual and tabular features
        # Shape: (B, Feature_Dim + 32)
        combined = torch.cat([bag_embedding, meta_embedding], dim=1)

        # --- Step 4: Classification ---
        # Shape: (B, 1)
        logits = self.classifier(combined)

        return logits
