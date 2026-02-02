import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUDecoupledInteraction(nn.Module):
    """
    Implements the Stabilized GLU-Decoupled Structural Injection module.

    Logic:
    1. Gather paired states h_j.
    2. Zero-mask unpaired states (h_j = 0).
    3. Compute GLU message based ONLY on h_j (Decoupled).
    4. Compute Gate based on [h_i; h_j] via a wide, stabilized MLP.
    5. Inject message: h_out = LayerNorm(h_i + gate * message).
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(GLUDecoupledInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Generation: (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Decoupled from h_i.
        self.msg_content = nn.Linear(hidden_dim, hidden_dim)
        self.msg_gate = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate
        # Input: [h_i; h_j] -> 2 * hidden_dim
        # Projects to hidden_dim (wide) -> LayerNorm -> GELU -> Sigmoid
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor (Batch, Seq_Len, Hidden_Dim) - The hidden states h_i
            pair_indices: Tensor (Batch, Seq_Len) - Indices of paired bases
            pair_mask: Tensor (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, dim = x.shape

        # 1. Gather h_j
        # Expand indices to match hidden dim: (B, L, D)
        # pair_indices is (B, L), we need to gather along dim 1
        flat_indices = pair_indices.view(batch_size, seq_len, 1).expand(-1, -1, dim)
        h_j = torch.gather(x, 1, flat_indices)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0.
        # pair_mask is (B, L), expand to (B, L, 1) for broadcasting
        mask = pair_mask.view(batch_size, seq_len, 1)
        h_j = h_j * mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired bases (h_j=0), this becomes bias terms, acting as loop embedding.
        content = self.msg_content(h_j)
        sigmoid_gate = torch.sigmoid(self.msg_gate(h_j))
        m_ij = content * sigmoid_gate

        # 4. Wide Stabilized MLP Gate
        # Input: Concatenate [h_i; h_j]
        cat_input = torch.cat([x, h_j], dim=-1)

        # Wide Projection -> LayerNorm -> GELU -> Sigmoid
        z_raw = self.gate_proj(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.gate_out(z_act))

        # 5. Injection & Post-Normalization
        # h_res = h_i + g_ij * m_ij
        update = g_ij * m_ij
        update = self.dropout(update)
        h_res = x + update
        h_out = self.final_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Stabilized GLU-Decoupled BiGRU Model.

    Structure:
    1. 1D Conv Stem
    2. 4 Layers of BiGRU (Hidden 384 per direction -> 768 total)
    3. Interleaved GLUDecoupledInteraction modules (after layers 0, 1, 2)
    4. Linear Output Head
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # ================= Hyperparameters =================
        input_dim = Config.INPUT_DIM
        conv_filters = Config.CONV_FILTERS
        kernel_size = Config.KERNEL_SIZE

        # BiGRU Hidden Dim (per direction)
        gru_hidden = Config.HIDDEN_DIM
        # Total hidden dimension after concatenation
        self.hidden_dim = gru_hidden * 2

        num_layers = Config.NUM_LAYERS
        dropout = Config.DROPOUT
        num_targets = 5

        # ================= Architecture =================

        # 1. Convolutional Stem
        # Projects (B, L, Input_Dim) -> (B, L, Conv_Filters)
        # Note: Conv1d expects (B, Channels, Length), so we permute in forward
        self.stem_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.stem_act = nn.GELU()
        self.stem_proj = nn.Linear(conv_filters, self.hidden_dim)
        self.stem_norm = nn.LayerNorm(self.hidden_dim)
        self.stem_dropout = nn.Dropout(dropout)

        # 2. Backbone (BiGRUs + Interactions)
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        for i in range(num_layers):
            # BiGRU Layer
            # Note: We use batch_first=True
            gru = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Interaction Layer (except after the final block)
            if i < num_layers - 1:
                interaction = GLUDecoupledInteraction(
                    hidden_dim=self.hidden_dim, dropout=dropout
                )
                self.interaction_layers.append(interaction)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, num_targets)

    def forward(self, inputs, pair_indices, pair_mask):
        """
        Args:
            inputs: (Batch, Seq_Len, Input_Dim)
            pair_indices: (Batch, Seq_Len)
            pair_mask: (Batch, Seq_Len)
        Returns:
            outputs: (Batch, Seq_Len, 5)
        """
        # ================= Stem =================
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Project to backbone dimension
        x = self.stem_proj(x)
        x = self.stem_norm(x)
        x = self.stem_dropout(x)

        # ================= Backbone =================
        # Iterate through layers
        for i, gru_layer in enumerate(self.gru_layers):
            # Apply BiGRU
            # GRU output: (B, L, 2*Hidden)
            gru_out, _ = gru_layer(x)

            # Residual connection around GRU is often helpful,
            # but standard RNN blocks usually just pass through.
            # Given "High-Capacity", we let the GRU transform the state fully.
            # However, to maintain gradient flow in deep networks, a residual
            # connection for the recurrent block itself is good practice.
            # Let's add a residual connection + Norm if dimensions match (they do here).
            x = x + gru_out
            # Optional: LayerNorm after GRU residual?
            # The strategy doesn't explicitly mandate it inside the block,
            # but the interaction module has a final LayerNorm.
            # We will proceed with x = x + gru_out.

            # Apply Interaction Module (if not last layer)
            if i < len(self.interaction_layers):
                interaction = self.interaction_layers[i]
                x = interaction(x, pair_indices, pair_mask)

        # ================= Head =================
        outputs = self.head(x)

        return outputs
