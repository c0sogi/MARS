import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Internally-Normalized Channel-Gated Interaction Module.

    Implements the structural injection mechanism with:
    1. Point-to-Point Gathering of neighbor states.
    2. Zero-Masking for unpaired bases.
    3. Non-Linear Message transformation.
    4. Internally-Normalized Gating (LayerNorm on logits).
    5. Residual Injection.
    6. Post-Normalization (LayerNorm on output).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message transformation: W_msg
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism: W_gate
        # Input is concatenation of [h_i; h_j] -> 2 * hidden_dim
        self.w_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Internal Normalization for gate logits (Stabilization)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Post-update Normalization (Stabilization)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, indices, mask):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Hidden_Dim).
            indices (torch.Tensor): Neighbor indices (Batch, Seq_Len).
            mask (torch.Tensor): Pair mask (Batch, Seq_Len), 1.0 if paired, 0.0 otherwise.

        Returns:
            torch.Tensor: Updated features (Batch, Seq_Len, Hidden_Dim).
        """
        batch_size, seq_len, dim = x.shape

        # 1. Gather neighbor states h_j
        # We expand indices to (B, L, D) to gather along the sequence dimension
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)
        h_j = torch.gather(x, 1, indices_expanded)

        # 2. Zero-Masking
        # Ensure unpaired positions (where mask is 0) contribute zero vector
        mask_expanded = mask.unsqueeze(-1)
        h_j = h_j * mask_expanded

        # 3. Non-Linear Message
        # m_{ij} = GELU(W_{msg} * h_j)
        m_ij = F.gelu(self.w_msg(h_j))

        # 4. Internally-Normalized Gate
        # z_{ij} = W_{gate} * [h_i; h_j]
        concat = torch.cat([x, h_j], dim=-1)
        z_ij = self.w_gate(concat)

        # Apply LayerNorm to logits before Sigmoid to prevent saturation
        z_hat_ij = self.gate_norm(z_ij)
        g_ij = torch.sigmoid(z_hat_ij)

        # 5. Injection (Residual Update)
        # h_{res} = h_i + g_{ij} * m_{ij}
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class SDIN_CG_BiGRU(nn.Module):
    """
    Stabilized Deep Internally-Normalized Channel-Gated BiGRU.

    Architecture:
    - Input: (N, 107, 14)
    - Stem: Conv1d (14 -> 256)
    - Backbone: 4 Blocks
        - Blocks 1-3: BiGRU (384 hidden) -> InteractionModule
        - Block 4: BiGRU (384 hidden)
    - Head: Linear (768 -> 5)
    """

    def __init__(self):
        super().__init__()

        # Dimensions from Config
        self.input_dim = Config.INPUT_DIM
        self.conv_dim = 256  # Fixed as per strategy description
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.gru_output_dim = self.hidden_dim * 2  # Bidirectional -> 768
        self.output_dim = Config.OUTPUT_DIM

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embedding, aggregates local k-mers
        self.conv_stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_dim,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )

        # 2. Deep Backbone (4 Blocks)

        # Block 1
        self.gru1 = nn.GRU(
            input_size=self.conv_dim,
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

        # Block 3
        self.gru3 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.inter3 = InteractionModule(self.gru_output_dim)

        # Block 4 (Final Block - No Interaction)
        self.gru4 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, self.output_dim)

    def forward(self, features, indices, mask):
        """
        Args:
            features (torch.Tensor): (Batch, Seq_Len, 14)
            indices (torch.Tensor): (Batch, Seq_Len)
            mask (torch.Tensor): (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 5)
        """
        # Permute for Conv1d: (B, L, 14) -> (B, 14, L)
        x = features.permute(0, 2, 1)

        # Stem
        x = self.conv_stem(x)
        x = F.gelu(x)

        # Permute back for RNN: (B, 256, L) -> (B, L, 256)
        x = x.permute(0, 2, 1)

        # Block 1
        x, _ = self.gru1(x)
        x = self.inter1(x, indices, mask)
        x = self.dropout(x)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, indices, mask)
        x = self.dropout(x)

        # Block 3
        x, _ = self.gru3(x)
        x = self.inter3(x, indices, mask)
        x = self.dropout(x)

        # Block 4
        x, _ = self.gru4(x)
        x = self.dropout(x)

        # Output Head
        out = self.head(x)

        return out
