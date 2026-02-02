import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StabilizedDecoupledInteraction(nn.Module):
    """
    Implements the Stabilized Decoupled Structural Interaction Module.

    Key Features:
    1. Point-to-Point Gather: Retrieves paired hidden state h_j.
    2. Zero-Masking: Explicitly forces h_j = 0 if unpaired.
    3. Bias-Driven Refinement: Unpaired bases generate a learnable bias embedding via GELU(bias).
    4. Stabilized MLP Gate: Uses Internal LayerNorm to prevent saturation.
    5. Post-Normalization: Applies LayerNorm after the residual connection.
    """

    def __init__(self, hidden_dim):
        super(StabilizedDecoupledInteraction, self).__init__()

        # Message Generation
        # Decoupled: Only takes h_j (or 0) as input.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stabilized MLP Gate
        # Input: [h_i; h_j]
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, mask):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Hidden_Dim).
            adj (torch.Tensor): Adjacency indices (Batch, Seq_Len).
            mask (torch.Tensor): Pair mask (Batch, Seq_Len). 1.0 if paired, 0.0 if unpaired.
        """
        B, L, H = x.shape

        # 1. Gather Context (h_j)
        # Expand adjacency indices to match hidden dimension
        adj_expanded = adj.unsqueeze(-1).expand(-1, -1, H)
        # Gather: x_pair[b, i, :] = x[b, adj[b, i], :]
        x_pair = torch.gather(x, 1, adj_expanded)

        # 2. Input Zero-Masking
        # If unpaired, mask is 0. Force x_pair to 0.
        # This ensures strictly no self-loop information flows through the pair channel
        # and activates the bias-driven refinement mechanism.
        mask_expanded = mask.unsqueeze(-1)
        x_pair = x_pair * mask_expanded

        # 3. Decoupled Message (Bias-Refined)
        # m = GELU(W * x_pair + b)
        # If x_pair is 0, m = GELU(b) -> Learnable loop embedding
        m = F.gelu(self.msg_proj(x_pair))

        # 4. Stabilized MLP Gate
        # Concatenate current state and pair state
        gate_in = torch.cat([x, x_pair], dim=-1)

        # Internal Normalization pipeline
        z_raw = self.gate_proj1(gate_in)
        z_norm = self.gate_norm(z_raw)  # Normalize internal activations
        z_act = F.gelu(z_norm)

        # Logit Projection (No normalization on logits to allow saturation)
        logits = self.gate_proj2(z_act)
        g = torch.sigmoid(logits)

        # 5. Injection
        h_res = x + g * m

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HCSDBR_BiGRU(nn.Module):
    """
    High-Capacity Stabilized Decoupled Bias-Refined BiGRU (HC-SDBR-BiGRU).

    Structure:
    - 1D Convolutional Stem
    - 4 Blocks of High-Capacity BiGRU (768 dim)
    - Interleaved Stabilized Decoupled Interaction Modules (Layers 1-3)
    - Linear Output Head
    """

    def __init__(self):
        super(HCSDBR_BiGRU, self).__init__()

        # Configuration
        self.input_dim = Config.INPUT_DIM
        self.stem_filters = Config.STEM_FILTERS
        self.stem_kernel = Config.STEM_KERNEL_SIZE

        # Backbone dims
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.total_hidden = self.hidden_dim * 2  # 768 (Bidirectional)
        self.num_layers = Config.NUM_LAYERS  # 4
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embeddings and aggregates local context
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.stem_filters,
                self.stem_kernel,
                padding=self.stem_kernel // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone Layers
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        current_dim = self.stem_filters

        for i in range(self.num_layers):
            # Bidirectional GRU
            # We maintain high capacity throughout the network
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)
            current_dim = self.total_hidden

            # Interaction Module
            # Applied after GRU in all blocks except the final one
            if i < self.num_layers - 1:
                interaction = StabilizedDecoupledInteraction(current_dim)
                self.interaction_layers.append(interaction)
            else:
                self.interaction_layers.append(None)

        # 3. Output Head
        self.head = nn.Linear(current_dim, self.num_targets)

    def forward(self, inputs, adjacency, mask):
        """
        Args:
            inputs (torch.Tensor): (Batch, Seq_Len, 14)
            adjacency (torch.Tensor): (Batch, Seq_Len)
            mask (torch.Tensor): (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.transpose(1, 2)
        x = self.stem(x)
        # Permute back: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Pass through backbone
        for i in range(self.num_layers):
            # BiGRU
            x, _ = self.gru_layers[i](x)

            # Interaction (if exists for this layer)
            if self.interaction_layers[i] is not None:
                x = self.interaction_layers[i](x, adjacency, mask)

        # Projection
        out = self.head(x)

        return out
