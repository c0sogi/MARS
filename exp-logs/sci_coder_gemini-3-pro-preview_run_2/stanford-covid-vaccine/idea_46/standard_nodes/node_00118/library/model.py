import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SmoothDenseBlock(nn.Module):
    """
    A dense block with dilated convolution, LayerNorm, and SiLU activation.
    Follows the structure:
    Conv(k=3, d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        # Standard Dilated Convolution
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        # LayerNorm is applied over channels, so we init with growth_rate
        self.ln1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # Pointwise Convolution
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (B, C_in, L)
        out = self.conv1(x)  # (B, Growth, L)

        # Permute for LayerNorm: (B, Growth, L) -> (B, L, Growth)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = out.permute(0, 2, 1)  # Back to (B, Growth, L)

        out = self.act1(out)

        out = self.conv2(out)  # (B, Growth, L)

        # Permute for LayerNorm
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)

        out = self.act2(out)
        out = self.dropout(out)

        # Dense Connection
        return torch.cat([x, out], dim=1)


class DenseTCN(nn.Module):
    """
    A stack of SmoothDenseBlocks with exponentially increasing dilation rates.
    """

    def __init__(self, in_channels, growth_rate, dilations):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels
        for d in dilations:
            block = SmoothDenseBlock(current_channels, growth_rate, d)
            self.blocks.append(block)
            current_channels += growth_rate
        self.out_channels = current_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SDFRNModel(nn.Module):
    """
    Smooth-Activated Dense-Feedback Recurrent Network (SDF-RN).
    Features:
    - Static Dense Dilated Backbone
    - Smooth Dense Feedback Module for recycled predictions
    - Augmented Gather for Partner Interactions
    - Compact Bi-GRU Aggregation
    """

    def __init__(self):
        super().__init__()

        # 1. Input Representation (Embeddings)
        self.seq_emb = nn.Embedding(4, Config.EMBED_DIM)
        self.struct_emb = nn.Embedding(3, Config.EMBED_DIM)
        self.loop_emb = nn.Embedding(7, Config.EMBED_DIM)
        self.partner_emb = nn.Embedding(5, Config.EMBED_DIM)  # 4 bases + 1 'None' token

        # 2. Static Backbone
        # Input: Sequence + Structure + Loop + Partner Identity
        backbone_in_dim = Config.EMBED_DIM * 4
        self.backbone = DenseTCN(
            backbone_in_dim, Config.BACKBONE_GROWTH_RATE, Config.BACKBONE_LAYERS
        )
        # Project to Latent Dim
        self.to_latent = nn.Conv1d(
            self.backbone.out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # 3. Feedback Module
        # Input: Recycled Targets (5) + Topology Embeddings (Struct + Loop)
        feedback_in_dim = 5 + Config.EMBED_DIM * 2
        self.feedback_net = DenseTCN(
            feedback_in_dim, Config.FEEDBACK_GROWTH_RATE, Config.FEEDBACK_LAYERS
        )
        # Project to Feedback Dim
        self.to_fb_emb = nn.Conv1d(
            self.feedback_net.out_channels, Config.FEEDBACK_DIM, kernel_size=1
        )

        # 4. Interaction & Aggregation
        # Input to RNN: (Latent + Feedback) for Self AND Partner
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2
        self.rnn = nn.GRU(
            rnn_input_dim, Config.RNN_HIDDEN, batch_first=True, bidirectional=True
        )

        # 5. Head
        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward(self, seq, struct, loop, partner_id, partner_idx, prev_pred=None):
        """
        Args:
            seq, struct, loop, partner_id: (B, L) LongTensors
            partner_idx: (B, L) LongTensor (Adjacency)
            prev_pred: (B, L, 5) or (B, 5, L) FloatTensor, optional
        """
        # --- 1. Static Representation ---
        # (B, L, Emb)
        x_seq = self.seq_emb(seq)
        x_struct = self.struct_emb(struct)
        x_loop = self.loop_emb(loop)
        x_part = self.partner_emb(partner_id)

        # Concatenate and Permute to (B, C, L)
        x_static = torch.cat([x_seq, x_struct, x_loop, x_part], dim=-1).permute(0, 2, 1)

        # Run Backbone
        z = self.backbone(x_static)
        z = self.to_latent(z)  # (B, Latent, L)

        # --- 2. Feedback Representation ---
        if prev_pred is None:
            prev_pred = torch.zeros(seq.size(0), 5, Config.SEQ_LEN, device=seq.device)
        else:
            # Ensure shape (B, 5, L)
            if prev_pred.shape[-1] != Config.SEQ_LEN:
                prev_pred = prev_pred.permute(0, 2, 1)

        # Prepare Topology Features for Feedback (Struct + Loop embeddings)
        # Permute to (B, Emb, L)
        topo_struct = x_struct.permute(0, 2, 1)
        topo_loop = x_loop.permute(0, 2, 1)

        # Concatenate: [Targets, Struct, Loop]
        fb_in = torch.cat([prev_pred, topo_struct, topo_loop], dim=1)

        # Run Feedback Net
        e_fb = self.feedback_net(fb_in)
        e_fb = self.to_fb_emb(e_fb)  # (B, FB_Dim, L)

        # --- 3. Interaction (Augmented Gather) ---
        # Combine Latent (Z) and Feedback (E_fb) -> Self Vector
        # (B, Latent+FB, L) -> (B, L, Latent+FB)
        h_self = torch.cat([z, e_fb], dim=1).permute(0, 2, 1)

        # Gather Partner Vector
        batch_size, seq_len, _ = h_self.shape

        # Create batch indices: (B, L)
        batch_indices = (
            torch.arange(batch_size, device=seq.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Handle unpaired bases in partner_idx (-1)
        # We clamp -1 to 0 to allow gathering, then mask the result later
        safe_partner_idx = partner_idx.clone()
        mask_unpaired = safe_partner_idx == -1
        safe_partner_idx[mask_unpaired] = 0

        # Gather: h_partner[b, i] = h_self[b, partner_idx[b, i]]
        h_partner = h_self[batch_indices, safe_partner_idx]  # (B, L, C)

        # Apply Null-Masking for unpaired bases
        h_partner[mask_unpaired] = 0.0

        # Fuse: Concat Self + Partner
        rnn_in = torch.cat([h_self, h_partner], dim=-1)  # (B, L, C*2)

        # --- 4. Aggregation & Head ---
        rnn_out, _ = self.rnn(rnn_in)
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
