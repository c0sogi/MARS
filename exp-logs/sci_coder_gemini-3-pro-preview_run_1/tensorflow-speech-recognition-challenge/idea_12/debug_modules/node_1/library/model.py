import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EnergyGatedAttentivePooling(nn.Module):
    """
    Implements the Energy-Gated Attention Mechanism.

    This layer computes a weighted average of temporal features, where the weights
    are derived from both the semantic features (spectrogram) and the absolute
    signal energy. This helps the model distinguish between 'silence' (low energy)
    and 'speech' (high energy) more effectively than spectral features alone.
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        # Project the 1D energy vector to the same dimension as the features
        self.energy_proj = nn.Conv1d(1, feat_dim, kernel_size=1)

        # Attention mechanism components
        # W_feat * f_t
        self.w_feat = nn.Conv1d(feat_dim, hidden_dim, kernel_size=1)
        # W_energy * e_t
        self.w_energy = nn.Conv1d(feat_dim, hidden_dim, kernel_size=1)
        # v^T * ...
        self.v = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x, energy):
        """
        Args:
            x (torch.Tensor): Feature map from backbone. Shape (B, C, T).
            energy (torch.Tensor): RMS energy vector. Shape (B, 1, T_orig).

        Returns:
            torch.Tensor: Global feature representation. Shape (B, C).
        """
        B, C, T = x.shape

        # 1. Align Energy Vector to Feature Map Temporal Resolution
        # The backbone downsamples the time dimension. We interpolate energy to match.
        # energy is (B, 1, T_orig)
        if energy.shape[-1] != T:
            energy_aligned = F.interpolate(
                energy, size=T, mode="linear", align_corners=False
            )
        else:
            energy_aligned = energy

        # 2. Project Energy to Feature Dimension
        # (B, 1, T) -> (B, C, T)
        e_proj = self.energy_proj(energy_aligned)

        # 3. Compute Attention Scores
        # Score = v^T * tanh(W_f * x + W_e * e)
        feat_attn = self.w_feat(x)  # (B, H, T)
        energy_attn = self.w_energy(e_proj)  # (B, H, T)

        # Additive combination
        combined = torch.tanh(feat_attn + energy_attn)

        # Project to scalar score per time step
        scores = self.v(combined)  # (B, 1, T)

        # 4. Compute Attention Weights
        alpha = F.softmax(scores, dim=2)  # (B, 1, T)

        # 5. Weighted Pooling
        # x: (B, C, T), alpha: (B, 1, T) -> (B, C)
        context = torch.sum(x * alpha, dim=2)

        return context


class EnergyGatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Energy-Gated Attentive Pooling.
    """

    def __init__(self, num_classes):
        super().__init__()

        # 1. Backbone: EfficientNet-B2
        # - in_chans=1: Adapts first conv for Log-Mel Spectrograms
        # - output_stride=16: Enforces dilated convolutions in the final stages
        #   (Stage 5) to preserve temporal resolution (stride 1, dilation 2).
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            output_stride=16,
            num_classes=0,  # Remove default classifier
            global_pool="",  # Remove default pooling
        )

        # Determine feature dimension (1408 for EfficientNet-B2)
        self.feat_dim = self.backbone.num_features

        # 2. Energy-Gated Pooling
        self.att_pool = EnergyGatedAttentivePooling(self.feat_dim)

        # 3. Classification Head
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        # 4. Dropout (Optional but recommended)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, x, energy):
        """
        Args:
            x (torch.Tensor): Log-Mel Spectrogram. Shape (B, 1, F, T).
            energy (torch.Tensor): Energy vector. Shape (B, 1, T_orig).

        Returns:
            torch.Tensor: Logits. Shape (B, num_classes).
        """
        # 1. Extract Features
        # Output shape: (B, C, F', T')
        features = self.backbone(x)

        # 2. Frequency Pooling
        # We average over the frequency dimension to get temporal feature sequences.
        # (B, C, F', T') -> (B, C, T')
        features = torch.mean(features, dim=2)

        # 3. Energy-Gated Attentive Pooling
        # Integrates the raw energy signal to guide attention
        embedding = self.att_pool(features, energy)

        # 4. Classification
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)

        return logits
