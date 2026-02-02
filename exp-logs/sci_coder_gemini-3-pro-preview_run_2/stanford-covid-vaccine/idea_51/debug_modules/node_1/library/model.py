import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LayerNormChannels(nn.Module):
    """
    Applies LayerNorm to channel dimension of (N, C, L) tensors.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        # (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)
        return x


class InputStem(nn.Module):
    """
    Decoupled Input Stem: Projects categorical inputs to dense space before normalization.
    Structure: Conv1d(k=1) -> LayerNorm -> SiLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = LayerNormChannels(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DenseDilatedBlock(nn.Module):
    """
    Pre-Activation Dense Dilated Block.
    Structure: LN -> SiLU -> Conv(k=3, d=d) -> LN -> SiLU -> Conv(k=1) -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.norm1 = LayerNormChannels(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        self.norm2 = LayerNormChannels(growth_rate)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act1(self.norm1(x))
        out = self.conv1(out)
        out = self.act2(self.norm2(out))
        out = self.conv2(out)
        out = self.drop(out)
        return out


class Backbone(nn.Module):
    """
    Pre-Activation Dense Dilated TCN Backbone.
    Accumulates features via dense connections.
    """

    def __init__(
        self,
        in_channels,
        growth_rate,
        layers,
        dilations,
        kernel_size,
        dropout,
        latent_dim,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for i in range(layers):
            dilation = dilations[i % len(dilations)]
            blk = DenseDilatedBlock(
                current_channels, growth_rate, kernel_size, dilation, dropout
            )
            self.blocks.append(blk)
            current_channels += growth_rate

        self.final_proj = nn.Conv1d(current_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        features = [x]
        for blk in self.blocks:
            # Dense connection: concatenate all previous outputs
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)

        total = torch.cat(features, dim=1)
        return self.final_proj(total)


class FeedbackModule(nn.Module):
    """
    Pure-Feedback Module: Lightweight Dense TCN for processing recycled predictions.
    """

    def __init__(self, in_channels, growth_rate, dilations, kernel_size, embed_dim):
        super().__init__()
        # Initial projection
        self.embedding = nn.Conv1d(in_channels, growth_rate, kernel_size=1)

        self.blocks = nn.ModuleList()
        current_channels = growth_rate

        for dilation in dilations:
            blk = DenseDilatedBlock(
                current_channels, growth_rate, kernel_size, dilation, dropout=0.1
            )
            self.blocks.append(blk)
            current_channels += growth_rate

        self.final_proj = nn.Conv1d(current_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.embedding(x)
        features = [x]
        for blk in self.blocks:
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)
        total = torch.cat(features, dim=1)
        return self.final_proj(total)


class DSPFN(nn.Module):
    """
    Decoupled-Stem Pure-Feedback Network (DS-PFN).
    """

    def __init__(self):
        super().__init__()

        # 1. Input Stem
        self.stem = InputStem(Config.INPUT_CHANNELS, Config.BACKBONE_GROWTH_RATE)

        # 2. Main Backbone
        self.backbone = Backbone(
            in_channels=Config.BACKBONE_GROWTH_RATE,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            layers=Config.BACKBONE_LAYERS,
            dilations=Config.BACKBONE_DILATIONS,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dropout=Config.BACKBONE_DROPOUT,
            latent_dim=Config.LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback = FeedbackModule(
            in_channels=Config.FEEDBACK_INPUT_CHANNELS,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=Config.FEEDBACK_DILATIONS,
            kernel_size=Config.FEEDBACK_KERNEL_SIZE,
            embed_dim=Config.FEEDBACK_EMBED_DIM,
        )

        # 4. Interaction & Aggregation (RNN)
        # Input size = (Latent + Feedback) * 2 (Self + Partner)
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
        )

        # 5. Output Head
        self.head = nn.Linear(Config.RNN_HIDDEN_SIZE * 2, 5)

        # Register unscored indices for strict masking
        # Total targets = 5. Scored indices are defined in Config.
        unscored = [i for i in range(5) if i not in Config.SCORED_INDICES]
        self.register_buffer(
            "unscored_indices", torch.tensor(unscored, dtype=torch.long)
        )

    def encode(self, inputs):
        """
        Pass 1: Static Feature Extraction.
        inputs: (N, C, L)
        Returns: Z (N, Latent, L)
        """
        x = self.stem(inputs)
        z = self.backbone(x)
        return z

    def decode(self, z, y_prev, partner_map):
        """
        Pass 2: Recurrent Feedback Loop.
        z: (N, Latent, L)
        y_prev: (N, 5, L) - Recycled predictions
        partner_map: (N, L) - Indices of paired bases
        """
        # 1. Strict Masking of Unscored Targets
        y_masked = y_prev.clone()
        y_masked[:, self.unscored_indices, :] = 0.0

        # 2. Compute Feedback Embeddings
        e_fb = self.feedback(y_masked)  # (N, 32, L)

        # 3. Interaction (Augmented Gather)
        # Construct Self Vector: [Z, E_fb]
        h_self = torch.cat([z, e_fb], dim=1)  # (N, 96, L)

        batch_size, channels, length = h_self.shape

        # Prepare partner map for gathering
        # Replace -1 (unpaired) with 0 temporarily for gather safety
        pm = partner_map.clone()
        mask_unpaired = pm == -1
        pm[mask_unpaired] = 0

        # Expand map to match channel dimension: (N, C, L)
        pm_expanded = pm.unsqueeze(1).expand(-1, channels, -1)

        # Gather Partner Vector: [Z_pair, E_fb_pair]
        h_partner = torch.gather(h_self, 2, pm_expanded)

        # Zero out vectors for unpaired bases
        h_partner = h_partner.masked_fill(mask_unpaired.unsqueeze(1), 0.0)

        # Fuse Self and Partner
        rnn_in = torch.cat([h_self, h_partner], dim=1)  # (N, 192, L)

        # 4. Global Aggregation (RNN)
        rnn_in = rnn_in.permute(0, 2, 1)  # (N, L, C)
        rnn_out, _ = self.rnn(rnn_in)

        # 5. Prediction Head
        logits = self.head(rnn_out)

        return logits.permute(0, 2, 1)  # (N, 5, L)

    def forward(self, inputs, partner_map, y_prev=None):
        """
        Standard forward pass (useful for inference wrapper or simple training).
        """
        z = self.encode(inputs)
        if y_prev is None:
            y_prev = torch.zeros(
                (inputs.size(0), 5, inputs.size(2)), device=inputs.device
            )
        return self.decode(z, y_prev, partner_map)
