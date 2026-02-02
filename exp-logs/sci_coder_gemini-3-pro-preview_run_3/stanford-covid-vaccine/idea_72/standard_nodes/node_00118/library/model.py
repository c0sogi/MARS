import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PointwiseFFN(nn.Module):
    """
    Pointwise Feed-Forward Network with Residual Connection and LayerNorm.
    Expands the feature dimension to digest structural updates.
    Structure: Linear -> GELU -> Dropout -> Linear
    """

    def __init__(self, hidden_dim, ffn_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Residual connection followed by normalization
        return self.norm(x + self.net(x))


class GLUInteractionModule(nn.Module):
    """
    Decoupled GLU-Structural Module.
    Performs neighbor gathering, zero-masking, GLU message computation, and gated injection.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Message Computation Layers: (W_c h_j + b_c) and (W_g h_j + b_g)
        # These operate on the gathered neighbor features.
        self.w_content = nn.Linear(hidden_dim, hidden_dim)
        self.w_gate_msg = nn.Linear(hidden_dim, hidden_dim)

        # Full-Rank Stabilized Gate: Projects h_i to a gate g_ij
        # Structure: Linear -> LayerNorm -> GELU -> Linear -> Sigmoid
        # This gate modulates how much of the structural message is accepted.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Final LayerNorm after injection
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, bpp_indices, bpp_masks):
        """
        Args:
            x: Input tensor (Batch, Seq, Hidden)
            bpp_indices: Indices of paired bases (Batch, Seq)
            bpp_masks: Mask indicating paired status (Batch, Seq), 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, hidden_dim = x.size()

        # 1. Gather neighbor features h_j
        # Expand indices to match hidden dimension: (B, L, H)
        flat_indices = bpp_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        # Gather along sequence dimension (dim=1)
        h_j = torch.gather(x, 1, flat_indices)

        # 2. Input Zero-Masking
        # If unpaired, mask is 0.0, forcing h_j to 0 vector.
        # Mask shape: (B, L, 1)
        mask = bpp_masks.unsqueeze(-1)
        h_j = h_j * mask

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # When h_j is 0 (unpaired), this becomes b_c * sigmoid(b_g), acting as a learned loop embedding.
        msg_content = self.w_content(h_j)
        msg_gate = torch.sigmoid(self.w_gate_msg(h_j))
        m_ij = msg_content * msg_gate

        # 4. Gated Injection
        # Gate g_ij is computed from the current node features x (h_i)
        g_ij = self.gate_mlp(x)

        # Injection: h_struct = LayerNorm(h_in + g_ij * m_ij)
        out = x + g_ij * m_ij
        return self.norm(out)


class EncoderBlock(nn.Module):
    """
    High-Capacity Encoder Block consisting of:
    1. BiGRU (Sequential Processing)
    2. GLU Interaction Module (Structural Injection)
    3. Pointwise FFN (Deep Refinement)
    """

    def __init__(self, input_dim, hidden_dim, ffn_dim, dropout):
        super().__init__()

        # BiGRU: Hidden dim is per direction. Output dim is hidden_dim * 2.
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # The output dimension of the BiGRU (e.g., 384 * 2 = 768)
        total_hidden = hidden_dim * 2

        # Structural Interaction
        self.glu_module = GLUInteractionModule(total_hidden)

        # Feed-Forward Network
        self.ffn = PointwiseFFN(total_hidden, ffn_dim, dropout)

        # Dropout for the recurrent output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bpp_indices, bpp_masks):
        # 1. BiGRU
        # gru_out: (Batch, Seq, Hidden*2)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)

        # 2. GLU Structural Injection
        glu_out = self.glu_module(gru_out, bpp_indices, bpp_masks)

        # 3. Pointwise FFN
        out = self.ffn(glu_out)

        return out


class RNAModel(nn.Module):
    """
    High-Capacity FFN-Enhanced GLU-BiGRU Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.INPUT_CHANNELS,
                config.CNN_FILTERS,
                kernel_size=config.CNN_KERNEL,
                padding=config.CNN_KERNEL // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone (Stacked Encoder Blocks)
        self.blocks = nn.ModuleList()

        current_input_dim = config.CNN_FILTERS
        hidden_dim = config.HIDDEN_DIM  # 384 per direction
        ffn_dim = config.FFN_DIM  # 1536
        dropout = config.DROPOUT

        for _ in range(config.NUM_LAYERS):
            block = EncoderBlock(current_input_dim, hidden_dim, ffn_dim, dropout)
            self.blocks.append(block)
            # The output of the block is bidirectional (hidden_dim * 2)
            current_input_dim = hidden_dim * 2

        # 3. Output Head
        # Projects final hidden state to the 5 target values
        self.head = nn.Linear(current_input_dim, config.NUM_TARGETS)

    def forward(self, inputs, bpp_indices, bpp_masks):
        """
        Args:
            inputs: (Batch, Seq, Channels)
            bpp_indices: (Batch, Seq)
            bpp_masks: (Batch, Seq)
        """
        # Permute for Conv1d: (Batch, Channels, Seq)
        x = inputs.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)

        # Permute back: (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)

        # Apply Blocks
        for block in self.blocks:
            x = block(x, bpp_indices, bpp_masks)

        # Apply Head
        out = self.head(x)

        return out
