import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Global Attention Pooling Layer.
    Aggregates spatial/temporal features weighted by a learned attention map.
    Input: (Batch, Channels, H, W)
    Output: (Batch, Channels)
    """

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, C, N) where N = H*W
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # Calculate attention scores
        # (B, N, C) -> (B, N, 1)
        attn_scores = self.conv(x)

        # Softmax over the spatial/temporal dimension N
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)

        # Weighted sum: sum(Features * Weights)
        # (B, N, C) * (B, N, 1) -> (B, N, C) -> Sum over N -> (B, C)
        out = torch.sum(x * attn_weights, dim=1)

        return out


class SiameseEquivariantNet(nn.Module):
    """
    Siamese Equivariant Dual-Stream Network.
    Stream A: Siamese EfficientNet for 4 anatomical EEG chains (Detail).
    Stream B: EfficientNet for 10-minute Spectrogram (Context).
    Fusion: Gated mechanism where Context modulates Detail.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ==========================
        # 1. Siamese EEG Stream
        # ==========================
        # Shared backbone for the 4 anatomical views.
        # Input shape per view: (Batch, 5, H, W)
        # in_chans=5 corresponds to the 5 electrodes in a chain.
        self.eeg_backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=5,
            num_classes=0,
            global_pool="",  # Disable default pooling to use AttentionPooling
        )

        # Feature dimension (e.g., 1280 for EfficientNet-B0)
        self.feature_dim = self.eeg_backbone.num_features

        # Attention Pooling to focus on specific time segments
        self.eeg_pool = AttentionPooling(self.feature_dim)

        # ==========================
        # 2. Context Spec Stream
        # ==========================
        # Backbone for the global 10-minute spectrogram.
        # Input shape: (Batch, 4, H, W)
        # in_chans=4 corresponds to the 4 regions (LL, RL, LP, RP).
        self.spec_backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=4,
            num_classes=0,
            global_pool="",
        )

        # Standard Global Average Pooling for context
        self.spec_pool = nn.AdaptiveAvgPool2d(1)

        # ==========================
        # 3. Fusion & Head
        # ==========================
        # The Detail feature is the concatenation of the 4 Siamese outputs
        self.detail_dim = 4 * self.feature_dim

        # Projection layer to map Context features to the Detail dimension for gating
        self.context_gate_proj = nn.Linear(self.feature_dim, self.detail_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROP_RATE)

        # Final Classification Head
        self.classifier = nn.Linear(self.detail_dim, Config.NUM_CLASSES)

    def forward(self, x_eeg: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_eeg: Tensor of shape (Batch, 4, 5, H, W)
            x_spec: Tensor of shape (Batch, 4, H, W)
        Returns:
            Logits of shape (Batch, Num_Classes)
        """
        batch_size = x_eeg.size(0)

        # ---------------------------
        # Stream A: Siamese Processing
        # ---------------------------
        # Collapse Batch and View dimensions: (B*4, 5, H, W)
        x_eeg_flat = x_eeg.view(batch_size * 4, 5, x_eeg.size(3), x_eeg.size(4))

        # Pass through shared backbone
        feat_eeg_map = self.eeg_backbone(x_eeg_flat)  # (B*4, C, H', W')

        # Apply Attention Pooling
        feat_eeg_vec = self.eeg_pool(feat_eeg_map)  # (B*4, C)

        # Reshape back to separate views: (B, 4, C)
        # Flatten to form the Detail vector: (B, 4*C)
        feat_detail = feat_eeg_vec.view(batch_size, -1)

        # ---------------------------
        # Stream B: Context Processing
        # ---------------------------
        # Pass through context backbone
        feat_spec_map = self.spec_backbone(x_spec)  # (B, C, H', W')

        # Apply Global Average Pooling
        feat_spec_vec = self.spec_pool(feat_spec_map).flatten(1)  # (B, C)

        # ---------------------------
        # Fusion: Gated Mechanism
        # ---------------------------
        # Project Context to generate the Gate: (B, 4*C)
        # Sigmoid ensures values are between 0 and 1
        gate = torch.sigmoid(self.context_gate_proj(feat_spec_vec))

        # Modulate Detail features with the Context Gate
        # Element-wise multiplication
        feat_fused = feat_detail * gate

        # ---------------------------
        # Classification
        # ---------------------------
        feat_fused = self.dropout(feat_fused)
        logits = self.classifier(feat_fused)

        return logits


def get_model(pretrained: bool = True) -> nn.Module:
    """Factory function to instantiate the model."""
    return SiameseEquivariantNet(pretrained=pretrained)
