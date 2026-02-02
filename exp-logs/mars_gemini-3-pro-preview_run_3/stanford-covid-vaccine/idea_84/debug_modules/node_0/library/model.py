import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StabilizedInteractionModule(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    Implements:
    1. Point-to-Point Gathering with Zero-Masking for unpaired bases.
    2. Decoupled GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
       - No h_i concatenation in message generation.
       - Unpaired bases (h_j=0) result in a pure learned bias vector.
    3. Stabilized MLP Gate:
       - Input: [h_i; h_j]
       - Wide Projection -> LayerNorm -> GELU -> Sigmoid.
    4. Residual Injection with Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled GLU Message components
        # W_c and W_g operate only on h_j (neighbor)
        self.w_content = nn.Linear(hidden_dim, hidden_dim)
        self.w_gate = nn.Linear(hidden_dim, hidden_dim)

        # Stabilized MLP Gate components
        # Input is concatenation of h_i and h_j
        self.gate_proj_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, SeqLen, HiddenDim)
            pair_indices: Tensor of shape (Batch, SeqLen) with indices of paired bases.
                          Unpaired bases should have index -1.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Gather h_j (neighbor features)
        # Handle -1 indices by replacing them with 0 temporarily for gathering, then masking
        valid_mask = (pair_indices != -1).unsqueeze(-1)  # (B, L, 1)
        safe_indices = pair_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)

        # Gather
        x_pair = torch.gather(x, 1, gather_indices)

        # Zero-masking for unpaired bases: if unpaired, h_j = 0
        x_pair = x_pair * valid_mask.float()

        # 2. Decoupled GLU Message
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Note: When x_pair is 0 (unpaired), this becomes bias_c * sigmoid(bias_g)
        msg_content = self.w_content(x_pair)
        msg_gate = torch.sigmoid(self.w_gate(x_pair))
        message = msg_content * msg_gate

        # 3. Stabilized MLP Gate
        # Input: [h_i; h_j]
        gate_input = torch.cat([x, x_pair], dim=-1)

        # Wide Projection -> Internal Norm -> GELU
        z_raw = self.gate_proj_in(gate_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Final Sigmoid Gate (No Logit Norm to allow saturation)
        gate = torch.sigmoid(self.gate_proj_out(z_act))

        # 4. Injection and Post-Normalization
        # h_res = h_i + g_ij * m_ij
        h_res = x + (gate * message)
        h_out = self.out_norm(h_res)

        return h_out


class MCSDBiGRU(nn.Module):
    """
    Massive-Capacity Stabilized Decoupled BiGRU (MC-SD-BiGRU).

    Architecture:
    1. 1D Convolutional Stem (Projection to backbone width).
    2. 4-Layer Backbone:
       - BiGRU (Hidden=512 per dir, Total=1024).
       - Stabilized Interaction Module.
    3. Linear Output Head.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM  # 512
        self.backbone_dim = self.hidden_dim * 2  # 1024 (BiGRU output)
        self.num_layers = Config.NUM_LAYERS
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse input (14 channels) to dense embedding (512 channels)
        # Matches Config.CNN_FILTERS
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=Config.CNN_FILTERS,
                kernel_size=Config.CNN_KERNEL_SIZE,
                padding=Config.CNN_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.LayerNorm(
                Config.CNN_FILTERS
            ),  # Channel-wise LN usually requires permute, handled in forward
        )

        # 2. Massive-Capacity Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        current_dim = Config.CNN_FILTERS  # Starts at 512

        for _ in range(self.num_layers):
            # BiGRU: Input current_dim -> Output hidden_dim * 2 (1024)
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Interaction Module operates on the BiGRU output (1024 dim)
            interact = StabilizedInteractionModule(self.backbone_dim)
            self.interaction_layers.append(interact)

            # Next layer input is 1024
            current_dim = self.backbone_dim

        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(self.backbone_dim, self.num_targets)

    def forward(self, features, pair_indices):
        """
        Args:
            features: (Batch, SeqLen, 14)
            pair_indices: (Batch, SeqLen)
        """
        # 1. Stem
        # Conv1d expects (B, C, L)
        x = features.permute(0, 2, 1)
        x = self.stem[0](x)  # Conv
        x = self.stem[1](x)  # GELU

        # Permute back for LayerNorm and GRU: (B, L, C)
        x = x.permute(0, 2, 1)
        x = self.stem[2](x)  # LayerNorm

        # 2. Backbone
        for gru, interact in zip(self.gru_layers, self.interaction_layers):
            # BiGRU
            # x shape: (B, L, input_dim) -> (B, L, 1024)
            x, _ = gru(x)

            # Stabilized Interaction
            # x shape: (B, L, 1024) -> (B, L, 1024)
            x = interact(x, pair_indices)

            # Dropout between blocks
            x = self.dropout(x)

        # 3. Head
        out = self.head(x)

        return out
