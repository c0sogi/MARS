import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class NeighborhoodAttention(nn.Module):
    """
    Structural Neighborhood Attention Module.

    For each position i paired with j, this module attends to the window of hidden states
    around j (e.g., [j-1, j, j+1]) and adds the context to h_i via a gated residual.
    """

    def __init__(self, input_dim, window_size=3):
        super(NeighborhoodAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.half_window = window_size // 2

        # Learnable gating scalar for residual connection.
        # Initialized to 0 so the model starts with pure sequence processing
        # and gradually learns to incorporate structural info.
        self.gate = nn.Parameter(torch.zeros(1))

        # Scale factor for dot product attention
        self.scale = input_dim**-0.5

    def forward(self, x, pair_indices):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases, -1 if unpaired.

        Returns:
            torch.Tensor: Same shape as x, with structural context added.
        """
        B, L, D = x.shape
        device = x.device

        # 1. Prepare Neighbor Indices
        # We want to gather [j-1, j, j+1] for every i where pair_indices[i] == j.

        # Create a mask for paired bases (1 if paired, 0 if unpaired)
        # pair_indices is -1 for unpaired.
        mask = (pair_indices != -1).float().unsqueeze(-1)  # (B, L, 1)

        # Create safe indices: replace -1 with 0 to prevent index out of bounds during gather.
        # These positions will be masked out later, so the value 0 doesn't matter.
        safe_indices = pair_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Generate window offsets: e.g., [-1, 0, 1]
        offsets = torch.arange(
            -self.half_window, self.half_window + 1, device=device
        )  # (Window,)

        # Broadcast add offsets to indices
        # safe_indices: (B, L, 1)
        # offsets: (1, 1, Window)
        # neighbor_indices: (B, L, Window)
        neighbor_indices = safe_indices.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(
            0
        )

        # Clamp indices to [0, L-1] to handle boundaries (e.g. index 0 paired with 0 -> -1 is invalid)
        neighbor_indices = neighbor_indices.clamp(0, L - 1)

        # 2. Gather Keys/Values
        # We gather the hidden vectors from x corresponding to neighbor_indices.
        # We use advanced indexing: x[batch_idx, neighbor_indices]

        # Create batch index grid: (B, 1, 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1)

        # Gather: (B, L, Window, D)
        # keys_values[b, i, w, :] = x[b, neighbor_indices[b, i, w], :]
        keys_values = x[batch_idx, neighbor_indices]

        # 3. Compute Attention
        # Query: h_i -> (B, L, 1, D)
        query = x.unsqueeze(2)

        # Keys: keys_values -> (B, L, Window, D)
        # Compute Dot Product: Q * K^T
        # (B, L, 1, D) @ (B, L, D, Window) -> (B, L, 1, Window)
        scores = torch.matmul(query, keys_values.transpose(-2, -1))
        scores = scores * self.scale

        # Softmax over the window dimension
        attn_weights = F.softmax(scores, dim=-1)  # (B, L, 1, Window)

        # 4. Aggregate Context
        # Weights * Values
        # (B, L, 1, Window) @ (B, L, Window, D) -> (B, L, 1, D)
        context = torch.matmul(attn_weights, keys_values)
        context = context.squeeze(2)  # (B, L, D)

        # 5. Residual Connection with Gating
        # Only add context where the base is actually paired (mask).
        output = x + self.gate * (context * mask)

        return output


class StructuralBiGRU(nn.Module):
    """
    Main Model Architecture:
    1. 1D Convolutional Stem (Local Feature Extraction)
    2. BiGRU Layer 1
    3. Structural Neighborhood Attention (Injects 3D constraints)
    4. BiGRU Layer 2 & 3
    5. Linear Head
    """

    def __init__(self):
        super(StructuralBiGRU, self).__init__()

        # Load hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.conv_filters = Config.CONV_FILTERS
        self.conv_kernel = Config.CONV_KERNEL_SIZE
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_p = Config.DROPOUT
        self.attn_window = Config.ATTENTION_WINDOW
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embeddings and aggregates local k-mers.
        self.conv_stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)

        # 2. Recurrent Backbone
        # Layer 1: Conv Filters -> Hidden Dim
        self.gru1 = nn.GRU(
            input_size=self.conv_filters,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Structural Attention Mechanism
        # BiGRU output dimension is hidden_dim * 2
        self.gru_output_dim = self.hidden_dim * 2
        self.attention = NeighborhoodAttention(
            input_dim=self.gru_output_dim, window_size=self.attn_window
        )

        # Layer 2: Hidden Dim * 2 -> Hidden Dim
        self.gru2 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Layer 3: Hidden Dim * 2 -> Hidden Dim
        self.gru3 = nn.GRU(
            input_size=self.gru_output_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, self.num_targets)

    def forward(self, x, pair_indices):
        """
        Args:
            x: (Batch, Seq_Len, 14) - Input features
            pair_indices: (Batch, Seq_Len) - Structure map

        Returns:
            out: (Batch, Seq_Len, 5) - Predicted degradation rates
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # Stem
        x = self.conv_stem(x)
        x = self.act(x)
        x = self.dropout(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # GRU Layer 1
        x, _ = self.gru1(x)
        x = self.dropout(x)

        # Structural Injection
        # We attend to the neighbors of the paired base
        x = self.attention(x, pair_indices)

        # GRU Layer 2
        x, _ = self.gru2(x)
        x = self.dropout(x)

        # GRU Layer 3
        x, _ = self.gru3(x)
        x = self.dropout(x)

        # Head
        out = self.head(x)

        return out
