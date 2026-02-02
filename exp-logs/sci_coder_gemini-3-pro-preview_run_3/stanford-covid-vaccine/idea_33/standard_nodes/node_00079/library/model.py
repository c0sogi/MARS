import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputNormGatedInteraction(nn.Module):
    """
    Implements the Input-Normalized Channel-Gated Structural Interaction Module.
    Key features:
    - Point-to-point message passing based on secondary structure.
    - Zero-masking for unpaired bases.
    - Input Normalization for the gating mechanism.
    - Normalized Message Pathway.
    - Post-update Layer Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message Pathway: Transform neighbor state h_j
        # m_ij = LayerNorm(GELU(W_msg * h_j))
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_norm = nn.LayerNorm(hidden_dim)

        # Gating Pathway: Determine how much structure to inject
        # Input is concatenation of [h_i, h_j]
        # We apply LayerNorm on the INPUT to the gate projection for stability
        self.gate_norm_in = nn.LayerNorm(2 * hidden_dim)
        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Post-update Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_map):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim) - The sequence hidden states
            adj_map: (Batch, Seq_Len) - Indices of paired bases, -1 if unpaired
        """
        B, L, D = x.shape

        # 1. Gather Neighbor States (h_j)
        # Handle -1 indices (unpaired) by temporarily mapping them to 0
        # We will mask the result later so the value at index 0 doesn't matter for unpaired positions
        safe_indices = adj_map.clone()
        safe_indices[adj_map == -1] = 0

        # Create a mask for paired positions (1 if paired, 0 if unpaired)
        mask = (adj_map != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Expand indices for gathering: (B, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather h_j
        h_j = torch.gather(x, 1, gather_indices)

        # Apply Zero-Masking: Force h_j to 0 where unpaired
        h_j = h_j * mask

        # 2. Normalized Message Path
        m_ij = self.msg_proj(h_j)
        m_ij = F.gelu(m_ij)
        m_ij = self.msg_norm(m_ij)

        # 3. Input-Normalized Gating
        h_i = x
        # Concatenate current state and neighbor state
        x_gate = torch.cat([h_i, h_j], dim=-1)  # (B, L, 2D)

        # Apply LayerNorm to the inputs of the gate projection
        x_gate_norm = self.gate_norm_in(x_gate)

        # Calculate gate values
        g_ij = torch.sigmoid(self.gate_proj(x_gate_norm))

        # 4. Injection and Post-Normalization
        # Residual update
        h_res = h_i + g_ij * m_ij

        # Final normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNANet(nn.Module):
    """
    Deep Input-Normalized Channel-Gated BiGRU Architecture.
    Consists of:
    1. Convolutional Stem (1D Conv)
    2. Deep Backbone (4 Layers of BiGRU + Interaction)
    3. Output Head
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.num_layers = Config.NUM_LAYERS
        self.hidden_dim = Config.HIDDEN_DIM  # This is the hidden size of the GRU
        self.input_channels = Config.NUM_FEATURES
        self.conv_filters = Config.FILTERS
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout = Config.DROPOUT

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs (14 channels) to dense embedding (256 channels)
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Backbone
        self.gru_blocks = nn.ModuleList()
        self.interaction_blocks = nn.ModuleList()

        # BiGRU outputs 2 * hidden_dim features
        gru_out_dim = 2 * self.hidden_dim

        for i in range(self.num_layers):
            # First layer input comes from Stem (conv_filters), others from previous BiGRU/Interaction (gru_out_dim)
            input_dim = self.conv_filters if i == 0 else gru_out_dim

            # BiGRU Layer
            self.gru_blocks.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=self.hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Interaction Module is applied in all blocks EXCEPT the final one
            if i < self.num_layers - 1:
                self.interaction_blocks.append(InputNormGatedInteraction(gru_out_dim))

        # 3. Output Head
        # Projects final hidden state to 5 target values
        self.head = nn.Linear(gru_out_dim, 5)

    def forward(self, x, adj_map):
        """
        Args:
            x: (Batch, Seq_Len, 14) - Input features
            adj_map: (Batch, Seq_Len) - Adjacency map for structure
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Apply Backbone Blocks
        for i in range(self.num_layers):
            # BiGRU
            # GRU returns (output, h_n), we only need output
            x, _ = self.gru_blocks[i](x)

            # Apply Interaction Module if it exists for this layer
            if i < len(self.interaction_blocks):
                x = self.interaction_blocks[i](x, adj_map)

        # Apply Head
        out = self.head(x)

        return out
