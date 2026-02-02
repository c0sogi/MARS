import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Implements the Full-Rank Decoupled Structural Interaction.

    Mechanism:
    1. Gather neighbor state h_j.
    2. Mask h_j if unpaired (forcing h_j = 0).
    3. Compute Message: m_ij = GELU(W_msg * h_j + b_msg).
       Note: If unpaired, this becomes GELU(b_msg), acting as a learnable loop embedding.
    4. Compute Gate: Full-Rank MLP on [h_i; h_j].
    5. Update: h_new = LayerNorm(h_i + gate * message).
    """

    def __init__(self, hidden_dim, gate_hidden_dim):
        super(StructuralInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message Projection
        # Input: h_j (hidden_dim) -> Output: m_ij (hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate MLP (Full-Rank)
        # Input: [h_i; h_j] (2 * hidden_dim) -> Hidden: gate_hidden_dim -> Output: hidden_dim
        self.gate_in = nn.Linear(2 * hidden_dim, gate_hidden_dim)
        self.gate_norm = nn.LayerNorm(gate_hidden_dim)
        self.gate_out = nn.Linear(gate_hidden_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h (torch.Tensor): Current hidden states of shape (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases, shape (Batch, Seq_Len).
            pair_masks (torch.Tensor): 1.0 if paired, 0.0 if unpaired, shape (Batch, Seq_Len).

        Returns:
            torch.Tensor: Updated hidden states.
        """
        B, L, H = h.shape

        # 1. Gather h_j (Neighbor states)
        # Expand indices to match hidden dim: (B, L, H)
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, dim=1, index=idx_expanded)

        # 2. Apply Zero-Masking for unpaired bases
        # mask shape: (B, L, 1)
        mask_expanded = pair_masks.unsqueeze(-1)
        h_j = h_j * mask_expanded

        # 3. Compute Decoupled Message
        # m_ij = GELU(W * h_j + b)
        # For unpaired bases (h_j=0), this learns a bias vector (loop embedding).
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Compute Full-Rank Gate
        # Concatenate h_i and h_j: (B, L, 2*H)
        cat_input = torch.cat([h, h_j], dim=-1)

        # MLP: Linear -> Norm -> GELU -> Linear -> Sigmoid
        z = self.gate_in(cat_input)
        z = self.gate_norm(z)  # Internal Normalization
        z = F.gelu(z)
        logits = self.gate_out(z)
        g_ij = torch.sigmoid(logits)  # No Logit Norm, allow saturation

        # 5. Injection and Residual Connection
        update = g_ij * m_ij
        h_res = h + update

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Full-Rank Decoupled BiGRU.

    Architecture:
    1. Conv1d Stem (Projects inputs).
    2. 4-Layer Backbone:
       - Layers 1-3: BiGRU -> StructuralInteractionModule.
       - Layer 4: BiGRU.
    3. Linear Output Head.
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # Hyperparameters from Config
        self.input_channels = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.kernel_size = Config.KERNEL_SIZE
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.gate_hidden_dim = Config.GATE_HIDDEN_DIM
        self.dropout_p = Config.DROPOUT
        self.num_targets = 5

        # 1. Convolutional Stem
        # Input: (B, C, L) -> Output: (B, F, L)
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.dropout_p)

        for i in range(self.num_layers):
            # First layer takes conv output, others take hidden_dim
            input_dim = self.conv_filters if i == 0 else self.hidden_dim

            # BiGRU: Output size is hidden_dim (hidden_dim // 2 * 2)
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_dim // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Interaction Module is applied after blocks 0, 1, 2 (not the final block 3)
            if i < self.num_layers - 1:
                interaction = StructuralInteractionModule(
                    hidden_dim=self.hidden_dim, gate_hidden_dim=self.gate_hidden_dim
                )
                self.interaction_layers.append(interaction)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, features, pair_indices, pair_masks):
        """
        Args:
            features (torch.Tensor): Input features (B, L, 14).
            pair_indices (torch.Tensor): Structure pair indices (B, L).
            pair_masks (torch.Tensor): Structure pair masks (B, L).

        Returns:
            torch.Tensor: Predictions (B, L, 5).
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = features.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)

        # Permute back for RNN: (B, F, L) -> (B, L, F)
        x = x.permute(0, 2, 1)

        # Apply Backbone Blocks
        for i in range(self.num_layers):
            # BiGRU
            # x input: (B, L, input_dim)
            # x output: (B, L, hidden_dim)
            x, _ = self.gru_layers[i](x)

            # Structural Interaction (if applicable for this block)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_masks)

            # Dropout between blocks
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # Apply Output Head
        out = self.head(x)

        return out
