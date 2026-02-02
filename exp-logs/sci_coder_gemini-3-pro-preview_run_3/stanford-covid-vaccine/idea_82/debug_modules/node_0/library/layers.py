import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ConvStem(nn.Module):
    """
    Convolutional Stem to project sparse inputs into dense embedding space.

    Structure:
    - Conv1d (kernel=3, padding=1)
    - GELU
    - Linear Projection to Hidden Dim
    - LayerNorm
    """

    def __init__(self, input_channels, stem_filters, hidden_dim, kernel_size=3):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=stem_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()
        self.proj = nn.Linear(stem_filters, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (N, L, C) -> Permute for Conv1d: (N, C, L)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # Permute back: (N, L, Filters)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.norm(x)
        return x


class GLURefinedInteraction(nn.Module):
    """
    GLU-Refined Decoupled Structural Interaction Module.

    Features:
    - Point-to-Point Gathering of neighbor features.
    - Explicit Zero-Masking for unpaired bases.
    - GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
    - Wide Stabilized MLP Gate: [h_i; h_j] -> Wide Proj -> LN -> GELU -> Proj -> Sigmoid
    - Residual Injection + Post-Normalization
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(GLURefinedInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Components
        self.fc_content = nn.Linear(hidden_dim, hidden_dim)
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate Components
        # Input is concatenation of h_i and h_j (2 * hidden_dim)
        # Wide hidden dimension is set to hidden_dim (preserving rank)
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_act = nn.GELU()
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, neighbor_indices, pair_masks):
        """
        Args:
            x: Tensor (N, L, D) - Current hidden states
            neighbor_indices: Tensor (N, L) - Indices of paired bases
            pair_masks: Tensor (N, L) - 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, dim = x.size()

        # 1. Gather Neighbor Features (h_j)
        # Expand indices to (N, L, D) for gathering
        idx_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, dim)
        # Gather: neighbor_features[b, i, k] = x[b, neighbor_indices[b, i], k]
        neighbor_features = torch.gather(x, 1, idx_expanded)

        # 2. Zero-Masking for Unpaired Bases
        # pair_masks is (N, L), unsqueeze to (N, L, 1)
        mask = pair_masks.unsqueeze(-1)
        neighbor_features = neighbor_features * mask

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired (h_j=0), this becomes b_c * sigmoid(b_g) (Learnable Bias)
        content = self.fc_content(neighbor_features)
        gate_signal = torch.sigmoid(self.fc_gate(neighbor_features))
        message = content * gate_signal

        # 4. Wide Stabilized MLP Gate
        # Concatenate h_i (x) and h_j (neighbor_features)
        cat_features = torch.cat([x, neighbor_features], dim=-1)

        z = self.gate_proj1(cat_features)
        z = self.gate_norm(z)  # Internal Normalization
        z = self.gate_act(z)
        logits = self.gate_proj2(z)
        gate_coeff = torch.sigmoid(logits)  # g_ij

        # 5. Injection and Post-Normalization
        # h_res = h_i + g_ij * m_ij
        injection = gate_coeff * message
        injection = self.dropout(injection)

        h_res = x + injection
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    Deep GLU-Refined Decoupled BiGRU Model.

    Architecture:
    1. ConvStem
    2. 4 Blocks of (BiGRU -> GLURefinedInteraction)
    3. Regression Head
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # Config parameters
        self.input_channels = Config.INPUT_CHANNELS
        self.stem_filters = Config.STEM_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # 1. Stem
        self.stem = ConvStem(
            input_channels=self.input_channels,
            stem_filters=self.stem_filters,
            hidden_dim=self.hidden_dim,
            kernel_size=Config.STEM_KERNEL_SIZE,
        )

        # 2. Backbone Layers
        # We define lists of layers.
        # Note: BiGRU hidden_size is half of total hidden_dim because it's bidirectional
        gru_hidden = self.hidden_dim // 2

        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()

        for i in range(self.num_layers):
            # Input dim for GRU:
            # Layer 0: Comes from Stem (hidden_dim)
            # Layer >0: Comes from previous Interaction (hidden_dim)
            self.grus.append(
                nn.GRU(
                    input_size=self.hidden_dim,
                    hidden_size=gru_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Interaction module follows GRU in each block
            # Except possibly the last one, but strategy says "except the final block"
            # However, usually refining the representation before the head is good.
            # The prompt says: "except the final block". We will follow strict instructions.
            if i < self.num_layers - 1:
                self.interactions.append(
                    GLURefinedInteraction(self.hidden_dim, dropout=self.dropout_rate)
                )
            else:
                # Placeholder to keep indexing simple, though we won't use it
                self.interactions.append(None)

        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, inputs, neighbor_indices, pair_masks):
        """
        Args:
            inputs: (N, L, 14)
            neighbor_indices: (N, L)
            pair_masks: (N, L)
        """
        # Stem
        x = self.stem(inputs)

        # Backbone
        for i in range(self.num_layers):
            # BiGRU
            # x is (N, L, D)
            x, _ = self.grus[i](x)

            # Apply Interaction if not the last layer
            if i < self.num_layers - 1:
                x = self.interactions[i](x, neighbor_indices, pair_masks)

            # Optional: Dropout between blocks (conservative)
            x = self.dropout(x)

        # Head
        out = self.head(x)
        return out
