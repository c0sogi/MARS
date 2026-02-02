import torch
import torch.nn as nn
from library.config import Config


class PointToPointAttention(nn.Module):
    """
    Sparse Point-to-Point Multi-Head Attention Module.

    This module implements a specialized attention mechanism constrained to the
    secondary structure graph of the RNA. For each nucleotide i, it attends
    ONLY to its paired nucleotide j (if it exists).

    Mechanism:
    1. Project inputs to Q, K, V.
    2. Gather K and V from the paired index j for each i.
    3. Compute attention score: Sigmoid(Scale * (Q_i . K_j)).
    4. Output: Score * V_j.
    5. Masking: If i is unpaired, the output is forced to 0.
    """

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        if self.head_dim * num_heads != input_dim:
            raise ValueError(
                f"Input dim {input_dim} must be divisible by num_heads {num_heads}"
            )

        # Projections
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)

        # Output projection
        self.w_out = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases, shape (Batch, Seq_Len).
                                         Values are indices [0, Seq_Len-1], or -1 if unpaired.

        Returns:
            torch.Tensor: Attended features of shape (Batch, Seq_Len, Hidden_Dim).
        """
        B, L, D = x.shape

        # 1. Linear Projections & Reshape to Heads
        # Shape: (B, L, H, D_head)
        q = self.w_q(x).view(B, L, self.num_heads, self.head_dim)
        k = self.w_k(x).view(B, L, self.num_heads, self.head_dim)
        v = self.w_v(x).view(B, L, self.num_heads, self.head_dim)

        # 2. Gather Neighbors (Sparse Attention)
        # Handle unpaired indices (-1) by temporarily mapping them to 0 to avoid gather errors.
        # We will apply a zero-mask later to suppress these positions.
        valid_mask = pair_indices != -1  # (B, L)
        safe_indices = pair_indices.clone()
        safe_indices[~valid_mask] = 0

        # Calculate flat indices for gathering across the batch
        # We want to gather from the same sample (batch index) but different sequence position.
        batch_offsets = torch.arange(B, device=x.device) * L
        flat_indices = safe_indices + batch_offsets.unsqueeze(1)  # (B, L)
        flat_indices = flat_indices.view(-1)  # Flatten to (B*L)

        # Flatten K and V for gathering: (B*L, H, D_head)
        k_flat = k.view(B * L, self.num_heads, self.head_dim)
        v_flat = v.view(B * L, self.num_heads, self.head_dim)

        # Gather: Select the K and V of the paired base
        k_neighbor = k_flat[flat_indices].view(B, L, self.num_heads, self.head_dim)
        v_neighbor = v_flat[flat_indices].view(B, L, self.num_heads, self.head_dim)

        # 3. Attention Calculation
        # Score = Dot(Q_i, K_j)
        # (B, L, H, D_head) * (B, L, H, D_head) -> sum last dim -> (B, L, H)
        score = torch.sum(q * k_neighbor, dim=-1) * self.scale

        # Non-linear compatibility (Sigmoid)
        attn_weights = torch.sigmoid(score)  # (B, L, H)

        # 4. Weighted Aggregation
        # Output = Weight * V_j
        # (B, L, H, 1) * (B, L, H, D_head) -> (B, L, H, D_head)
        out = attn_weights.unsqueeze(-1) * v_neighbor

        # Reshape back to (B, L, D)
        out = out.reshape(B, L, D)

        # 5. Output Projection
        out = self.w_out(out)

        # 6. Zero-Masking
        # Explicitly zero out outputs for unpaired positions
        mask_expanded = valid_mask.unsqueeze(-1).type_as(out)
        out = out * mask_expanded

        return self.dropout(out)


class SPMHABiGRU(nn.Module):
    """
    Sparse Point-to-Point Multi-Head Attention BiGRU.

    Architecture:
    1. Conv1d Stem (14 -> 256)
    2. 3 Blocks of [BiGRU -> PointToPointAttention (except last block)]
    3. Linear Head (-> 5 targets)
    """

    def __init__(self):
        super().__init__()

        # Config
        self.input_dim = Config.INPUT_DIM
        self.cnn_filters = Config.CNN_FILTERS
        self.cnn_kernel = Config.CNN_KERNEL
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_heads = Config.NUM_HEADS
        self.dropout_p = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.cnn_filters,
                kernel_size=self.cnn_kernel,
                padding=self.cnn_kernel // 2,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        # 2. Backbone
        self.gru_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        # BiGRU outputs 2 * hidden_dim
        gru_out_dim = self.hidden_dim * 2

        # Input dim for the first GRU is cnn_filters
        current_input_dim = self.cnn_filters

        for i in range(self.num_layers):
            # BiGRU
            self.gru_layers.append(
                nn.GRU(
                    input_size=current_input_dim,
                    hidden_size=self.hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Next layer input will be this layer's output
            current_input_dim = gru_out_dim

            # Structural Interaction (Attention) - applied after GRU, before next block
            # Except for the final block
            if i < self.num_layers - 1:
                self.attn_layers.append(
                    PointToPointAttention(
                        input_dim=gru_out_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout_p,
                    )
                )
                self.layer_norms.append(nn.LayerNorm(gru_out_dim))

        # 3. Head
        self.head = nn.Linear(gru_out_dim, self.num_targets)

    def forward(self, inputs, pair_indices, mask=None):
        """
        Args:
            inputs (torch.Tensor): (Batch, Seq_Len, 14)
            pair_indices (torch.Tensor): (Batch, Seq_Len)
            mask (torch.Tensor, optional): (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.transpose(1, 2)

        # Stem
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Backbone
        for i in range(self.num_layers):
            # BiGRU
            # x: (B, L, Input_Dim) -> (B, L, 2*Hidden)
            x, _ = self.gru_layers[i](x)

            # Structural Interaction (if applicable)
            if i < len(self.attn_layers):
                # Attention
                attn_out = self.attn_layers[i](x, pair_indices)

                # Residual + Norm
                x = self.layer_norms[i](x + attn_out)

        # Head
        out = self.head(x)

        return out
