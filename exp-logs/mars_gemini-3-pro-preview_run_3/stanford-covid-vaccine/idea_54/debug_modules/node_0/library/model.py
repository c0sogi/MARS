import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT


class DecoupledInteractionLayer(nn.Module):
    """
    Implements the Decoupled Structural Interaction Module with:
    1. Point-to-Point Gathering.
    2. Input Zero-Masking (Bias-Driven Refinement for loops).
    3. Stabilized MLP Gating (Internal LayerNorm).
    4. Post-Normalization.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message generation: m_ij = GELU(W * h_j + b)
        # When h_j is masked (0), m_ij = GELU(b), serving as a loop embedding.
        self.fc_msg = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Stabilized MLP Gate
        # Input: [h_i; h_j] -> Project -> LN -> GELU -> Project -> Sigmoid
        self.fc_gate_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm_gate = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.fc_gate_2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-normalization
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases.
            pair_masks: (Batch, Seq_Len, 1) - 1.0 if paired, 0.0 if unpaired.
        """
        B, L, D = x.shape

        # 1. Gather neighbor features h_j
        # Create batch indices for advanced indexing
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)

        # Gather: h_j = x[b, pair_indices[b, l]]
        h_j = x[batch_idx, pair_indices]  # (B, L, D)

        # 2. Input Zero-Masking
        # Strictly avoid self-loops and noise from unpaired indices.
        # If unpaired, pair_masks is 0, forcing h_j to 0 vector.
        h_j = h_j * pair_masks

        # 3. Decoupled Message
        m_ij = F.gelu(self.fc_msg(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate h_i (x) and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2D)

        z_raw = self.fc_gate_1(cat_input)
        z_norm = self.norm_gate(z_raw)  # Stabilize MLP internals
        z_act = F.gelu(z_norm)
        logits = self.fc_gate_2(z_act)
        g_ij = torch.sigmoid(logits)  # No logit normalization

        # 5. Injection
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.norm_out(h_res)

        return self.dropout(h_out)


class DeepStabilizedBiGRU(nn.Module):
    """
    Deep Stabilized Bias-Refined Decoupled BiGRU Architecture.
    Consists of a Conv1d stem, 4 BiGRU blocks, and decoupled structural interaction layers.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        # 1. Convolutional Stem
        # Projects sparse inputs (14 channels) to dense embedding (256 channels)
        self.stem_dim = 256
        self.conv_stem = nn.Conv1d(input_dim, self.stem_dim, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        self.interactions = nn.ModuleList()

        # 2. Deep Stabilized Backbone
        for i in range(num_layers):
            # Determine input size for the GRU
            # First layer: Stem (256) -> Hidden (384)
            # Others: Hidden (384) -> Hidden (384)
            in_size = self.stem_dim if i == 0 else hidden_dim

            # BiGRU: Output dimension is 2 * hidden_size.
            # We want the output to match hidden_dim (384), so hidden_size = 192.
            gru = nn.GRU(
                input_size=in_size,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.blocks.append(gru)

            # Decoupled Interaction Module
            # Applied after blocks 0, 1, 2. Not applied after the final block (3).
            if i < num_layers - 1:
                self.interactions.append(DecoupledInteractionLayer(hidden_dim, dropout))

        # 3. Output Head
        self.head = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_masks):
        # x: (Batch, Seq_Len, 14)

        # --- Stem ---
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv_stem(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Back to (B, L, C)

        # --- Backbone ---
        for i, gru in enumerate(self.blocks):
            # BiGRU
            x, _ = gru(x)

            # Interaction (if applicable for this block)
            if i < len(self.interactions):
                x = self.interactions[i](x, pair_indices, pair_masks)

        # --- Head ---
        logits = self.head(x)
        return logits
