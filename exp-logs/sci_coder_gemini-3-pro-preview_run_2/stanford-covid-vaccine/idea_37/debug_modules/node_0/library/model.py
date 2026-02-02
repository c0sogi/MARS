import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    EMBED_DIM,
    HIDDEN_DIM,
    LAYERS,
    KERNEL_SIZE,
    DILATIONS,
    DROPOUT,
    LATENT_DIM,
    COND_DIM,
    FEEDBACK_LAYERS,
    FEEDBACK_CHANNELS,
    TARGET_COLS,
    SCORED_COLS,
)


class DenseBlock(nn.Module):
    """
    Single Dilated Residual Block with Dense Connection.
    Input: (B, InCh, L)
    Output: (B, InCh + GrowthRate, L)
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        # Dense connection: Concatenate input and output
        return torch.cat([x, out], dim=1)


class DenseDilatedTCN(nn.Module):
    """
    Static Backbone: Stack of DenseBlocks with increasing dilation.
    """

    def __init__(self, in_dim, hidden_dim, layers, kernel_size, dilations, dropout):
        super(DenseDilatedTCN, self).__init__()
        self.entry_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        current_dim = hidden_dim
        # Growth rate is set to hidden_dim
        growth_rate = hidden_dim

        for i in range(layers):
            d = dilations[i] if i < len(dilations) else 1
            block = DenseBlock(
                in_channels=current_dim,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_dim += growth_rate

        self.out_dim = current_dim

    def forward(self, x):
        # x: (B, C, L)
        out = self.entry_conv(x)
        for block in self.blocks:
            out = block(out)
        return out


class ConditionalFeedbackModule(nn.Module):
    """
    Processes previous predictions conditioned on static structure features.
    Masks unscored columns before processing.
    """

    def __init__(self, latent_dim, cond_dim, out_channels, layers):
        super(ConditionalFeedbackModule, self).__init__()

        # Identify indices of scored columns for masking
        self.scored_indices = [
            i for i, col in enumerate(TARGET_COLS) if col in SCORED_COLS
        ]
        self.num_targets = len(TARGET_COLS)

        # Project static latent features to conditioning vector
        self.cond_proj = nn.Conv1d(latent_dim, cond_dim, kernel_size=1)

        # Input to TCN = Masked Preds + Condition
        in_channels = self.num_targets + cond_dim

        self.tcn_layers = nn.ModuleList()
        for i in range(layers):
            # Lightweight TCN with small dilations
            d = 2**i
            conv = nn.Conv1d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                dilation=d,
                padding=d,
            )
            self.tcn_layers.append(conv)

        self.act = nn.ReLU()

    def forward(self, preds, static_latent):
        # preds: (B, L, 5)
        # static_latent: (B, LatentDim, L)

        B, L, _ = preds.shape
        device = preds.device

        # 1. Masking Unscored Columns
        mask = torch.zeros(self.num_targets, device=device)
        mask[self.scored_indices] = 1.0
        # Broadcast multiply: (B, L, 5) * (5,)
        masked_preds = preds * mask

        # Transpose to (B, 5, L) for Conv1d
        masked_preds_t = masked_preds.permute(0, 2, 1)

        # 2. Conditioning
        cond = self.cond_proj(static_latent)  # (B, CondDim, L)

        # 3. Fusion
        x = torch.cat([masked_preds_t, cond], dim=1)  # (B, 5+CondDim, L)

        # 4. Processing
        for conv in self.tcn_layers:
            x = conv(x)
            x = self.act(x)

        return x  # (B, OutCh, L)


class SCR_DN(nn.Module):
    """
    Structure-Conditional Recurrent Dense Network.
    """

    def __init__(self):
        super(SCR_DN, self).__init__()

        # 1. Static Backbone
        self.backbone = DenseDilatedTCN(
            in_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            layers=LAYERS,
            kernel_size=KERNEL_SIZE,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        # Project backbone output to Latent Dimension Z
        self.latent_proj = nn.Conv1d(self.backbone.out_dim, LATENT_DIM, kernel_size=1)

        # 2. Conditional Feedback Module
        self.feedback_mod = ConditionalFeedbackModule(
            latent_dim=LATENT_DIM,
            cond_dim=COND_DIM,
            out_channels=FEEDBACK_CHANNELS,
            layers=FEEDBACK_LAYERS,
        )

        # 3. Interaction & Head
        # Node features = Static Z + Feedback
        self.node_dim = LATENT_DIM + FEEDBACK_CHANNELS

        # Interaction fuses Self + Partner
        # BiGRU Input = NodeDim * 2
        gru_input_dim = self.node_dim * 2

        # BiGRU Hidden = input // 2
        gru_hidden_dim = gru_input_dim // 2

        self.bigru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        self.head = nn.Linear(gru_hidden_dim * 2, len(TARGET_COLS))

    def forward(self, x, partner_indices, pairing_mask):
        # x: (B, L, EmbedDim)
        # partner_indices: (B, L)
        # pairing_mask: (B, L)

        B, L, _ = x.shape

        # Transpose input for Backbone: (B, EmbedDim, L)
        x_t = x.permute(0, 2, 1)

        # 1. Compute Static Latent Z (Once)
        backbone_out = self.backbone(x_t)
        z = self.latent_proj(backbone_out)  # (B, LatentDim, L)

        outputs = []

        # Initialize predictions with zeros for the first pass
        current_preds = torch.zeros(B, L, len(TARGET_COLS), device=x.device)

        # 2. Iterative Refinement Loop
        # Pass 1: Zero feedback -> Preds 1
        # Pass 2: Preds 1 (detached) -> Preds 2
        for i in range(2):
            if i > 0:
                preds_input = current_preds.detach()
            else:
                preds_input = current_preds

            # Generate Feedback Embedding
            feedback = self.feedback_mod(preds_input, z)  # (B, FeedbackCh, L)

            # Fuse Static Z and Feedback
            node_feat_t = torch.cat([z, feedback], dim=1)  # (B, NodeDim, L)
            node_feat = node_feat_t.permute(0, 2, 1)  # (B, L, NodeDim)

            # Interaction: Gather Partner Features
            # Create batch indices for gather
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)

            # Gather paired features
            partner_feat = node_feat[batch_idx, partner_indices]

            # Mask unpaired positions (multiply by 0)
            partner_feat = partner_feat * pairing_mask.unsqueeze(-1)

            # Concatenate Self + Partner
            combined = torch.cat([node_feat, partner_feat], dim=2)  # (B, L, NodeDim*2)

            # BiGRU Aggregation
            gru_out, _ = self.bigru(combined)

            # Prediction Head
            logits = self.head(gru_out)  # (B, L, 5)

            current_preds = logits
            outputs.append(current_preds)

        return outputs
