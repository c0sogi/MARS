import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Strict Output Masking.

    Implements the logic:
    1. Gather neighbor hidden states (h_j).
    2. Decoupled Message: m_ij = GELU(W_msg * h_j).
    3. Channel-Wise Gate: g_ij = Sigmoid(W_gate * [h_i; h_j]).
    4. Strict Output Masking: u_ij = (g_ij * m_ij) * M_pair.
    5. Residual + Post-Norm: h_out = LayerNorm(h_i + u_ij).
    """

    def __init__(self, hidden_dim):
        super(StructuralInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message generation (derived solely from neighbor to decouple)
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism (derived from joint context)
        self.w_gate = nn.Linear(2 * hidden_dim, hidden_dim)

        # Post-Normalization layer
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            bpp_indices: LongTensor of shape (Batch, Seq_Len). Indices of paired bases.
            bpp_mask: FloatTensor of shape (Batch, Seq_Len). 1.0 if paired, 0.0 otherwise.
        """
        batch_size, seq_len, hidden_dim = x.size()

        # 1. Gather Neighbor States
        # Expand indices to match hidden dimension: (B, L, H)
        expanded_indices = bpp_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        # Gather: neighbor_x[b, i, :] = x[b, bpp_indices[b, i], :]
        neighbor_x = torch.gather(x, 1, expanded_indices)

        # 2. Decoupled Message
        # Compute message using only the neighbor's state
        msg = F.gelu(self.w_msg(neighbor_x))

        # 3. Channel-Wise Gating
        # Compute gate using both current and neighbor states
        gate_input = torch.cat([x, neighbor_x], dim=-1)
        gate = torch.sigmoid(self.w_gate(gate_input))

        # 4. Compute Update with Strict Output Masking
        # u_ij = (g_ij * m_ij) * M_pair
        # The mask ensures unpaired bases receive exactly 0 update
        mask_broadcast = bpp_mask.unsqueeze(-1)
        update = (gate * msg) * mask_broadcast

        # 5. Residual Connection and Post-LayerNorm
        # Post-Norm stabilizes deep architectures
        out = self.layer_norm(x + update)

        return out


class DeepDecoupledBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.

    Architecture:
    1. Input (One-Hot Encoded Features)
    2. Convolutional Stem (1D Conv + GELU)
    3. 4 Blocks of BiGRU.
       - Blocks 1, 2, 3: BiGRU -> StructuralInteractionModule
       - Block 4: BiGRU only
    4. Linear Output Head
    """

    def __init__(self, config: Config):
        super(DeepDecoupledBiGRU, self).__init__()
        self.config = config

        # Architecture Hyperparameters
        input_dim = config.input_dim
        stem_dim = 256
        hidden_dim = config.hidden_dim  # 384
        gru_out_dim = hidden_dim * 2  # Bidirectional implies 2 * hidden_dim = 768

        # 1. Convolutional Stem
        # Projects sparse inputs into a dense embedding space and aggregates local k-mers
        self.stem_conv = nn.Conv1d(input_dim, stem_dim, kernel_size=3, padding=1)
        self.stem_act = nn.GELU()

        # 2. Deep Stabilized Backbone
        self.num_layers = config.num_layers  # 4

        self.gru_layers = nn.ModuleList()
        self.sim_layers = nn.ModuleList()

        for i in range(self.num_layers):
            # First layer takes stem output, others take previous GRU output
            in_size = stem_dim if i == 0 else gru_out_dim

            # BiGRU Layer
            gru = nn.GRU(
                input_size=in_size,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Structural Interaction Module
            # Applied to all blocks except the final one
            if i < self.num_layers - 1:
                sim = StructuralInteractionModule(gru_out_dim)
                self.sim_layers.append(sim)
            else:
                self.sim_layers.append(None)

        # 3. Output Head
        # Projects the final hidden state to the target classes
        self.head = nn.Linear(gru_out_dim, config.num_classes)

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Forward pass of the model.

        Args:
            x: Input tensor (Batch, Seq_Len, Input_Dim)
            bpp_indices: Neighbor indices (Batch, Seq_Len)
            bpp_mask: Binary mask for pairs (Batch, Seq_Len)

        Returns:
            logits: Predicted values (Batch, Seq_Len, Num_Classes)
        """
        # 1. Stem
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        # Permute back to (Batch, Seq_Len, Channels) for RNN
        x = x.permute(0, 2, 1)

        # 2. Backbone
        for i in range(self.num_layers):
            # BiGRU
            # GRU returns (output, h_n). We only use the sequence output.
            x, _ = self.gru_layers[i](x)

            # Structural Interaction (if applicable for this layer)
            if self.sim_layers[i] is not None:
                x = self.sim_layers[i](x, bpp_indices, bpp_mask)

        # 3. Head
        logits = self.head(x)

        return logits
