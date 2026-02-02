import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PermuteLayerNorm(nn.Module):
    """
    LayerNorm wrapper for (N, C, L) tensors.
    Permutes to (N, L, C), applies LayerNorm, and permutes back.
    """

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.ln(x)
        # (N, L, C) -> (N, C, L)
        return x.transpose(1, 2)


class SpatialStem(nn.Module):
    """
    Input Stem with Spatial Convolution (k=3) to capture local n-grams immediately.
    Replaces pointwise stems to fix the 'Input Stem Flaw'.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = PermuteLayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PostActivationBlock(nn.Module):
    """
    Post-Activation Block:
    Conv(k=3) -> Norm -> Act -> Conv(k=1) -> Norm -> Act -> Dropout
    Decouples spatial aggregation from channel mixing.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.spatial_norm = PermuteLayerNorm(out_channels)
        self.spatial_act = nn.SiLU()

        self.pointwise_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.pointwise_norm = PermuteLayerNorm(out_channels)
        self.pointwise_act = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.spatial_act(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_norm(x)
        x = self.pointwise_act(x)

        x = self.dropout(x)
        return x


class DenseBackbone(nn.Module):
    """
    Dense Dilated TCN Backbone.
    Each block receives concatenated outputs of all previous blocks.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = PostActivationBlock(
                current_dim, growth_rate, dilation=d, dropout=dropout
            )
            self.blocks.append(block)
            current_dim += growth_rate

        self.out_channels = current_dim

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            # Concatenate all previous features
            in_feat = torch.cat(features, dim=1)
            out_feat = block(in_feat)
            features.append(out_feat)

        # Return the concatenation of all features (Dense connection)
        return torch.cat(features, dim=1)


class FeedbackModule(nn.Module):
    """
    Pure-Feedback Module.
    Processes recycled predictions with a lightweight Dense TCN.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        super().__init__()
        # Initial projection
        self.embedding = nn.Conv1d(in_channels, growth_rate, kernel_size=1)

        # Lightweight backbone
        self.backbone = DenseBackbone(growth_rate, growth_rate, dilations, dropout)

        # Output projection
        self.out_proj = nn.Conv1d(
            self.backbone.out_channels, growth_rate * 2, kernel_size=1
        )  # Output 32 channels

    def forward(self, x):
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.out_proj(x)
        return x


class SSPFN(nn.Module):
    """
    Spatial-Stem Pure-Feedback Dense Network.
    """

    def __init__(self):
        super().__init__()

        # --- Hyperparameters ---
        self.seq_len = Config.SEQ_LENGTH
        input_dim = 18  # 4 (seq) + 3 (struct) + 7 (loop) + 4 (partner)

        stem_kernel = Config.STEM_KERNEL_SIZE
        hidden_dim = Config.HIDDEN_DIM
        growth_rate = Config.GROWTH_RATE
        dilations = Config.DILATIONS
        dropout = Config.DROPOUT
        latent_dim = Config.LATENT_DIM

        fb_growth_rate = Config.FEEDBACK_GROWTH_RATE
        rnn_hidden = Config.RNN_HIDDEN_DIM

        # --- 1. Input Stem (Spatial) ---
        self.stem = SpatialStem(input_dim, hidden_dim, kernel_size=stem_kernel)

        # --- 2. Main Backbone ---
        self.backbone = DenseBackbone(hidden_dim, growth_rate, dilations, dropout)

        # Project backbone output to latent Z
        self.z_proj = nn.Conv1d(self.backbone.out_channels, latent_dim, kernel_size=1)

        # --- 3. Pure-Feedback Module ---
        # Input is 5 targets.
        # We will mask unscored ones inside forward, but input dim is 5.
        self.feedback_module = FeedbackModule(
            5, fb_growth_rate, dilations[:4], dropout
        )  # Shorter depth for feedback
        self.fb_dim = fb_growth_rate * 2  # 32

        # --- 4. Interaction & Aggregation ---
        # Input to RNN: (Z + E_fb) for self + (Z + E_fb) for partner = (64+32) * 2 = 192
        fusion_dim = (latent_dim + self.fb_dim) * 2

        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Head
        self.head = nn.Linear(rnn_hidden * 2, 5)

        # Mask for feedback (Strict Masking)
        # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Unscored: deg_pH10(2), deg_50C(4)
        # Mask: [1, 1, 0, 1, 0]
        self.register_buffer(
            "feedback_mask",
            torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32).view(1, 5, 1),
        )

    def forward(self, x, partner_indices, feedback_input=None):
        """
        Args:
            x: (B, 18, L) Input features
            partner_indices: (B, L) Indices of paired bases
            feedback_input: (B, L, 5) Recycled predictions from previous step.
                            If None, initializes with zeros.
        Returns:
            pred: (B, L, 5)
        """
        B, C, L = x.shape

        # 1. Static Backbone Processing
        # -----------------------------
        h = self.stem(x)
        h = self.backbone(h)
        z = self.z_proj(h)  # (B, Latent=64, L)

        # 2. Feedback Processing
        # ----------------------
        if feedback_input is None:
            feedback_input = torch.zeros((B, L, 5), device=x.device, dtype=x.dtype)

        # Transpose feedback to (B, 5, L) for Conv1d
        fb_in = feedback_input.transpose(1, 2)

        # Strict Masking: Zero out unscored channels
        fb_in = fb_in * self.feedback_mask

        # Generate feedback embeddings
        e_fb = self.feedback_module(fb_in)  # (B, 32, L)

        # 3. Interaction (Augmented Gather)
        # ---------------------------------
        # Concatenate Z and E_fb: (B, 96, L)
        node_feat = torch.cat([z, e_fb], dim=1)

        # Gather partner features
        # partner_indices is (B, L). We need to gather from dim 2 of node_feat.
        # Create batch indices
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)

        # Handle -1 in partner_indices (unpaired)
        # Replace -1 with 0 temporarily for gathering, then mask result
        valid_mask = partner_indices != -1
        safe_indices = partner_indices.clone()
        safe_indices[~valid_mask] = 0

        # Gather: (B, 96, L) -> (B, L, 96) for gathering -> (B, 96, L)
        node_feat_t = node_feat.transpose(1, 2)  # (B, L, 96)
        partner_feat_t = node_feat_t[batch_idx, safe_indices]  # (B, L, 96)

        # Mask unpaired positions
        partner_feat_t = partner_feat_t * valid_mask.unsqueeze(-1).float()

        # Transpose back to (B, 96, L)
        partner_feat = partner_feat_t.transpose(1, 2)

        # Fusion: Concatenate Self and Partner (B, 192, L)
        fusion = torch.cat([node_feat, partner_feat], dim=1)

        # 4. Global Aggregation (RNN)
        # ---------------------------
        # Permute for RNN: (B, L, 192)
        rnn_in = fusion.transpose(1, 2)

        rnn_out, _ = self.gru(rnn_in)  # (B, L, Hidden*2)

        # 5. Head
        # -------
        pred = self.head(rnn_out)  # (B, L, 5)

        return pred
