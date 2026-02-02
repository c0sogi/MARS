import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ChannelGatedInteraction(nn.Module):
    """
    Implements the Zero-Masked Channel-Gating mechanism with Post-Normalization.

    Logic:
    1. Gather neighbor states based on secondary structure.
    2. Zero-mask unpaired neighbors to prevent noise injection.
    3. Compute non-linear messages.
    4. Compute channel-wise gates based on node pair context.
    5. Apply residual connection and Post-LayerNorm.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Message transformation: Transform neighbor state -> Message
        self.w_msg = nn.Linear(dim, dim)

        # Gating projection: Takes [h_i; h_j] -> Gate
        self.w_gate = nn.Linear(dim * 2, dim)

        # Post-Normalization (Critical for stability in deep hybrid networks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (B, L, D) - Sequence features from the backbone.
            pair_indices: (B, L) - Indices of paired bases (0-based).
            pair_masks: (B, L) - 1.0 if paired, 0.0 if unpaired.
        """
        B, L, D = x.shape

        # 1. Gather neighbor states
        # Expand indices to match the feature dimension: (B, L, D)
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather features: h_j for each i
        h_neighbor = torch.gather(x, 1, idx_expanded)

        # 2. Zero-Masking
        # If unpaired (mask=0), force neighbor state to 0 vector.
        # pair_masks is (B, L), expand to (B, L, 1)
        mask_expanded = pair_masks.unsqueeze(-1)
        h_neighbor = h_neighbor * mask_expanded

        # 3. Non-Linear Message
        # m_ij = GELU(W_msg * h_j)
        m = F.gelu(self.w_msg(h_neighbor))

        # 4. Channel-Wise Gating
        # g_ij = sigmoid(W_gate * [h_i; h_j])
        concat = torch.cat([x, h_neighbor], dim=-1)
        g = torch.sigmoid(self.w_gate(concat))

        # 5. Residual Injection
        # h_res = h_i + g_ij * m_ij
        h_res = x + g * m

        # 6. Post-Normalization
        h_out = self.norm(h_res)

        return h_out


class DeepBiGRUNet(nn.Module):
    """
    Deep Post-Norm BiGRU with Zero-Masked Channel-Gating.

    Architecture:
    - 1D Conv Stem
    - 4 Blocks of (BiGRU -> Interaction -> Dropout)
      (Note: The final block does not have the Interaction module)
    - Linear Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.cnn_filters = Config.CNN_FILTERS
        self.cnn_kernel = Config.CNN_KERNEL_SIZE
        self.dropout_rate = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Input is (B, L, 14), Conv1d expects (B, 14, L)
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.cnn_filters,
            kernel_size=self.cnn_kernel,
            padding=self.cnn_kernel // 2,
        )
        self.stem_act = nn.GELU()

        # 2. Deep Stabilized Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Track current dimension to handle transition from CNN to RNN
        current_dim = self.cnn_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            # We set hidden_size to hidden_dim // 2 so that the bidirectional output matches hidden_dim
            rnn = nn.GRU(
                input_size=current_dim,
                hidden_size=self.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(rnn)
            self.dropouts.append(nn.Dropout(self.dropout_rate))

            # Interaction Layer
            # Interleaved in the first N-1 blocks
            if i < self.num_layers - 1:
                self.interaction_layers.append(ChannelGatedInteraction(self.hidden_dim))
            else:
                self.interaction_layers.append(None)

            # Update dimension for next layer (BiGRU output is always hidden_dim)
            current_dim = self.hidden_dim

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (B, L, 14) - One-hot encoded features
            pair_indices: (B, L) - Structural adjacency
            pair_masks: (B, L) - Structural validity mask
        """
        # --- Stem ---
        # Permute for Conv1d: (B, 14, L)
        x = x.transpose(1, 2)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        # Permute back: (B, L, C)
        x = x.transpose(1, 2)

        # --- Backbone ---
        for i in range(self.num_layers):
            # GRU
            x, _ = self.gru_layers[i](x)
            x = self.dropouts[i](x)

            # Interaction (if exists for this block)
            if self.interaction_layers[i] is not None:
                x = self.interaction_layers[i](x, pair_indices, pair_masks)

        # --- Head ---
        # Project to 5 target values
        out = self.head(x)

        return out
