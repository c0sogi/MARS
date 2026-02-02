import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Stabilized MLP-Gated Decoupled Interaction Module.

    Implements the structural injection mechanism with:
    1. Point-to-Point Gather of paired states.
    2. Zero-Masking for unpaired bases.
    3. Decoupled Message generation (allowing loop embeddings via bias).
    4. Stabilized MLP Gating (LayerNorm on hidden gate activations).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message Projection
        # Bias=True is critical: when h_j is masked to 0 (unpaired),
        # the bias acts as a learnable 'loop embedding'.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Stabilized MLP Gate
        # Input: Concatenation of [h_i; h_j] -> 2 * hidden_dim
        self.gate_l1 = nn.Linear(2 * hidden_dim, hidden_dim)

        # Normalization applied strictly to the hidden layer of the gate
        # This stabilizes the MLP internals without destroying sparsity (Lesson 75/79)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Output projection for the gate
        self.gate_l2 = nn.Linear(hidden_dim, hidden_dim)

        # Final LayerNorm after residual injection
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h: Tensor (Batch, Seq, Hidden)
            pair_indices: LongTensor (Batch, Seq)
            pair_masks: Tensor (Batch, Seq) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = h.shape

        # 1. Gather paired states h_j
        # pair_indices contains values in [0, L-1].
        # We offset them by batch index to gather from the flattened batch.
        batch_offsets = torch.arange(B, device=h.device) * L
        flat_indices = pair_indices + batch_offsets.unsqueeze(1)  # (B, L)
        flat_indices = flat_indices.view(-1)

        flat_h = h.view(-1, D)
        h_j = flat_h[flat_indices].view(B, L, D)

        # 2. Zero-Masking
        # If unpaired (mask=0), force h_j to 0.
        mask = pair_masks.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. Decoupled Message
        # m_ij = GELU(W * h_j + b)
        # For unpaired bases, this becomes GELU(b).
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate context
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2*D)

        # MLP Hidden Layer -> Norm -> Act
        z_raw = self.gate_l1(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Logit Projection -> Sigmoid (No normalization on logits)
        logits = self.gate_l2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection & Residual
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    3-Layer Bidirectional GRU with Interleaved Stabilized MLP-Gated Structural Injection.

    Structure:
    - Conv1d Stem
    - Block 1: BiGRU -> InteractionModule
    - Block 2: BiGRU -> InteractionModule
    - Block 3: BiGRU (No Interaction)
    - Linear Head
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embedding space
        self.conv = nn.Conv1d(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.CONV_FILTERS,
            kernel_size=config.CONV_KERNEL_SIZE,
            padding=config.CONV_KERNEL_SIZE // 2,
        )

        # 2. Backbone
        self.hidden_dim = config.HIDDEN_DIM
        self.gru_output_dim = self.hidden_dim * 2  # Bidirectional

        # Block 1
        self.gru1 = nn.GRU(
            input_size=config.CONV_FILTERS,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter1 = InteractionModule(self.gru_output_dim)

        # Block 2
        self.gru2 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter2 = InteractionModule(self.gru_output_dim)

        # Block 3 (Final Block - No Interaction as per instructions)
        self.gru3 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, config.NUM_TARGETS)

    def forward(self, features, pair_indices, pair_masks):
        """
        Args:
            features: (B, L, 14)
            pair_indices: (B, L)
            pair_masks: (B, L)
        """
        # Permute for Conv1d: (B, C, L)
        x = features.transpose(1, 2)

        # Stem
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Permute back for GRU: (B, L, C)
        x = x.transpose(1, 2)

        # Block 1
        x, _ = self.gru1(x)
        x = self.inter1(x, pair_indices, pair_masks)
        x = self.dropout(x)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, pair_indices, pair_masks)
        x = self.dropout(x)

        # Block 3
        x, _ = self.gru3(x)
        x = self.dropout(x)

        # Head
        out = self.head(x)

        return out
