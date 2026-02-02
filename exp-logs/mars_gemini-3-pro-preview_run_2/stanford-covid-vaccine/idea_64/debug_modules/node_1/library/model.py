import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConvStem(nn.Module):
    """
    Spatial Stem: Standard Convolution (k=3) -> LayerNorm -> SiLU.
    Ensures immediate mixing of adjacent nucleotides.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvStem, self).__init__()
        # Padding ensures output length matches input length
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (Batch, Channels, Length)
        x = self.conv(x)

        # Permute for LayerNorm: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = self.act(x)

        # Permute back: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        return x


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Block:
    Dilated Conv (k=3) -> LN -> SiLU -> Pointwise Conv (k=1) -> LN -> SiLU -> Dropout.
    Decouples spatial aggregation from channel mixing.
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super(PostActDenseBlock, self).__init__()

        # 1. Spatial Aggregation (Dilated Conv)
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, dilation=dilation, padding=dilation
        )
        self.ln1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # 2. Channel Mixing (Pointwise Conv)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, InChannels, Length)

        # Block 1
        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        # Block 2
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        return out


class DenseTCN(nn.Module):
    """
    Hierarchical Dense Backbone.
    Layers consume outputs from all prior layers (DenseNet structure).
    """

    def __init__(self, in_channels, growth_rate, dilation_rates, latent_dim):
        super(DenseTCN, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilation_rates:
            blk = PostActDenseBlock(current_channels, growth_rate, d)
            self.blocks.append(blk)
            current_channels += growth_rate

        # Project concatenated features to latent dimension
        self.project = nn.Conv1d(current_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            in_feat = torch.cat(features, dim=1)
            out_feat = block(in_feat)
            features.append(out_feat)

        # Final concatenation and projection
        total_feat = torch.cat(features, dim=1)
        out = self.project(total_feat)
        return out


class FeedbackModule(nn.Module):
    """
    Global-Context Feedback Module.
    Processes recycled predictions with strict channel masking.
    """

    def __init__(self):
        super(FeedbackModule, self).__init__()

        # Identify indices to zero out (unscored targets)
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Scored: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Unscored to Mask: deg_pH10(2), deg_50C(4)
        self.register_buffer("mask_indices", torch.tensor([2, 4], dtype=torch.long))

        # Spatial Stem for Feedback
        # Input: 5 channels -> Output: Feedback Growth Rate (16)
        self.stem = ConvStem(
            Config.NUM_TARGETS, Config.FEEDBACK_GROWTH_RATE, kernel_size=3
        )

        # Lightweight Dense TCN
        self.backbone = DenseTCN(
            in_channels=Config.FEEDBACK_GROWTH_RATE,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilation_rates=Config.DILATION_RATES,
            latent_dim=Config.FEEDBACK_DIM,
        )

    def forward(self, prev_preds):
        # prev_preds: (Batch, Length, 5)

        # 1. Channel Masking
        # Clone to ensure we don't modify the input tensor in place
        x = prev_preds.clone()
        x[..., self.mask_indices] = 0.0

        # Permute to (Batch, Channels, Length) for Conv processing
        x = x.permute(0, 2, 1)

        # 2. Processing
        x = self.stem(x)
        x = self.backbone(x)  # Returns (B, FeedbackDim, L)

        # Return as (B, L, FeedbackDim)
        return x.permute(0, 2, 1)


class ML_GFN(nn.Module):
    """
    Masked-Loss Global-Feedback Network.
    Integrates Static TCN, Feedback Module, Partner Interaction, and RNN Aggregation.
    """

    def __init__(self):
        super(ML_GFN, self).__init__()

        # =====================================================================
        # 1. Static Input Processing
        # =====================================================================
        # Input channels: 18 (4 Seq + 3 Struct + 7 Loop + 4 PartnerID)
        # Stem projects to Growth Rate to match DenseBlock expectations
        self.static_stem = ConvStem(18, Config.GROWTH_RATE)

        self.static_backbone = DenseTCN(
            in_channels=Config.GROWTH_RATE,
            growth_rate=Config.GROWTH_RATE,
            dilation_rates=Config.DILATION_RATES,
            latent_dim=Config.LATENT_DIM,
        )

        # =====================================================================
        # 2. Feedback Module
        # =====================================================================
        self.feedback_module = FeedbackModule()

        # =====================================================================
        # 3. Interaction & Aggregation
        # =====================================================================
        # Input to Interaction: Z (64) + E_fb (32) = 96 per base
        # Input to RNN: Self (96) + Partner (96) = 192
        self.feature_dim = Config.LATENT_DIM + Config.FEEDBACK_DIM
        rnn_input_dim = self.feature_dim * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        # Final Projection: 2 * Hidden (Bidirectional) -> 5 Targets
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def encode_static(self, inputs):
        """
        Computes static embeddings Z from inputs.
        Can be called once per batch during training to save compute.
        """
        # inputs: (Batch, Length, 18)
        x = inputs.permute(0, 2, 1)  # (B, 18, L)
        x = self.static_stem(x)
        z = self.static_backbone(x)  # (B, 64, L)
        return z.permute(0, 2, 1)  # (B, L, 64)

    def decode_dynamic(self, z, pair_indices, prev_preds=None):
        """
        Computes final predictions given static embeddings and previous predictions.
        """
        # z: (Batch, Length, 64)
        # pair_indices: (Batch, Length)
        # prev_preds: (Batch, Length, 5) or None

        B, L, _ = z.shape
        device = z.device

        # Initialize prev_preds if None (First Pass)
        if prev_preds is None:
            prev_preds = torch.zeros((B, L, Config.NUM_TARGETS), device=device)

        # 1. Get Feedback Embeddings
        e_fb = self.feedback_module(prev_preds)  # (B, L, 32)

        # 2. Construct Self Vector
        self_feat = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

        # 3. Construct Partner Vector (Augmented Gather)
        # Handle -1 indices (unpaired) by pointing to 0 and then masking
        gather_indices = pair_indices.clone()
        unpaired_mask = gather_indices == -1
        gather_indices[unpaired_mask] = 0

        # Expand indices for gathering across feature dimension
        # gather_indices: (B, L) -> (B, L, 96)
        gather_indices_exp = gather_indices.unsqueeze(-1).expand(
            -1, -1, self_feat.size(2)
        )

        # Gather
        partner_feat = torch.gather(self_feat, 1, gather_indices_exp)

        # Apply Null-Masking for unpaired bases
        partner_feat[unpaired_mask] = 0.0

        # 4. Fusion
        combined = torch.cat([self_feat, partner_feat], dim=2)  # (B, L, 192)

        # 5. Global Aggregation (RNN)
        rnn_out, _ = self.rnn(combined)  # (B, L, 2*Hidden)

        # 6. Projection
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits

    def forward(self, inputs, pair_indices, prev_preds=None):
        """
        Full forward pass.
        """
        z = self.encode_static(inputs)
        return self.decode_dynamic(z, pair_indices, prev_preds)
