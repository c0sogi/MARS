import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConvStem(nn.Module):
    """
    Convolutional Stem: Projects sparse inputs into a dense embedding space.
    Input: (B, L, 14) -> Output: (B, L, 256)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL,
            padding=Config.CONV_KERNEL // 2,
        )
        self.act = nn.GELU()

    def forward(self, x):
        # Permute to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # Permute back to (B, L, C)
        x = x.permute(0, 2, 1)
        return x


class WideGatedInteraction(nn.Module):
    """
    Wide Stabilized Decoupled Structural Injection Module.
    Performs point-to-point message passing based on RNA secondary structure.
    """

    def __init__(self, hidden_dim, gate_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Projection: m_ij = GELU(W * h_j + b)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate
        # Input context: [h_i; h_j] -> Size: 2 * hidden_dim
        self.gate_in = nn.Linear(hidden_dim * 2, gate_dim)
        self.gate_norm = nn.LayerNorm(gate_dim)  # Internal Normalization
        self.gate_out = nn.Linear(gate_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_mask):
        """
        Args:
            h: Hidden states (B, L, H)
            pair_indices: Indices of paired bases (B, L)
            pair_mask: Mask indicating paired status (1.0 paired, 0.0 unpaired) (B, L)
        """
        B, L, H = h.shape

        # 1. Gather Neighbor States (h_j)
        # Expand indices to (B, L, H) to gather along the sequence dimension
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, 1, idx)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0. This ensures unpaired bases only see the bias term.
        mask = pair_mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. Decoupled Message Calculation
        # m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j=0), this becomes GELU(b_msg), a learnable loop embedding.
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Wide Stabilized MLP Gate
        # Context: Concatenate current state h_i and neighbor state h_j
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)

        # Wide Projection -> LayerNorm -> GELU
        z_wide = self.gate_in(cat_input)
        z_norm = self.gate_norm(z_wide)
        z_act = F.gelu(z_norm)

        # Output Projection -> Sigmoid
        logits = self.gate_out(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection and Residual Connection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HC_WG_BiGRU(nn.Module):
    """
    High-Capacity Wide-Gated Decoupled BiGRU.
    4-Layer Backbone with interleaved structural interaction modules.
    """

    def __init__(self):
        super().__init__()

        # 1. Convolutional Stem
        self.stem = ConvStem()

        # 2. Backbone
        self.num_layers = Config.NUM_LAYERS
        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Initial input size comes from Stem (256)
        input_size = Config.CONV_FILTERS
        hidden_size = Config.HIDDEN_DIM  # 384 per direction

        for i in range(self.num_layers):
            # Bidirectional GRU
            # Output size will be hidden_size * 2 = 768
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.grus.append(gru)

            # Calculate output size of this GRU layer
            gru_out_size = hidden_size * 2

            # Structural Interaction Module
            # Applied after every block EXCEPT the final block
            if i < self.num_layers - 1:
                interaction = WideGatedInteraction(
                    hidden_dim=gru_out_size,
                    gate_dim=Config.GATE_HIDDEN_DIM,  # 768 (Wide)
                )
                self.interactions.append(interaction)

            self.dropouts.append(nn.Dropout(Config.DROPOUT))

            # Input to the next layer is the output of the current layer
            input_size = gru_out_size

        # 3. Output Head
        self.head = nn.Linear(input_size, Config.NUM_TARGETS)

    def forward(self, features, pair_indices, pair_mask):
        """
        Args:
            features: (B, L, 14)
            pair_indices: (B, L)
            pair_mask: (B, L)
        """
        # Stem
        x = self.stem(features)

        # Backbone Layers
        for i in range(self.num_layers):
            # GRU
            x, _ = self.grus[i](x)

            # Dropout
            x = self.dropouts[i](x)

            # Interaction (except final block)
            if i < self.num_layers - 1:
                x = self.interactions[i](x, pair_indices, pair_mask)

        # Head
        out = self.head(x)

        return out
