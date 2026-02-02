import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AffineStructuralModule(nn.Module):
    """
    Implements the Affine Decoupled Structural Injection module.

    Logic:
    1. Gather neighbor states h_j.
    2. Zero-mask unpaired neighbors.
    3. Compute Affine Scale (s) and Shift (t) from h_j.
    4. Compute Gate (g) from [h_i; h_j].
    5. Update h_i with affine transformation: h_i * s + t.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Internal LayerNorms for stability before projections
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_g = nn.LayerNorm(hidden_dim * 2)  # Input is cat(h_i, h_j)

        # Projections for Scale, Shift, and Gate
        self.proj_s = nn.Linear(hidden_dim, hidden_dim)
        self.proj_t = nn.Linear(hidden_dim, hidden_dim)
        self.proj_g = nn.Linear(hidden_dim * 2, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases
            pair_masks: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, H = x.shape

        # 1. Gather neighbor states h_j
        # Create batch indices for gathering: (B, L)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)

        # Gather: x[b, pair_indices[b, l], :]
        # Note: pair_indices for unpaired bases point to a valid index (0), but will be masked out.
        h_neighbor = x[batch_idx, pair_indices]  # (B, L, H)

        # 2. Input Zero-Masking
        # Explicitly force h_j = 0 for unpaired bases
        mask = pair_masks.unsqueeze(-1)  # (B, L, 1)
        h_neighbor = h_neighbor * mask

        # 3. Affine Decoupled Messages
        # Apply LN before projection
        s_in = self.norm_s(h_neighbor)
        t_in = self.norm_t(h_neighbor)

        # Scale Factor: s = Tanh(Ws * h_j + bs)
        # Shift Factor: t = GELU(Wt * h_j + bt)
        # For unpaired bases (h_j=0), these become learned biases (Tanh(bs), GELU(bt)).
        s = torch.tanh(self.proj_s(s_in))
        t = F.gelu(self.proj_t(t_in))

        # 4. Context-Aware Gating
        # g = sigmoid(Wg * [h_i; h_j])
        cat_input = torch.cat([x, h_neighbor], dim=-1)  # (B, L, 2H)
        g_in = self.norm_g(cat_input)
        g = torch.sigmoid(self.proj_g(g_in))

        # 5. Affine Injection
        # u = (h_i * s) + t
        # This allows the structure to scale and shift the current features
        u = (x * s) + t

        # 6. Residual Update
        # h_res = h_i + g * u
        h_res = x + (g * u)

        # 7. Post-Normalization
        out = self.out_norm(h_res)

        return out


class DADBiGRUModel(nn.Module):
    """
    Deep Affine-Decoupled BiGRU (DAD-BiGRU)

    Architecture:
    - Conv1d Stem
    - 4-Layer Backbone:
        - Layer 1-3: BiGRU -> AffineStructuralModule
        - Layer 4: BiGRU
    - Linear Head
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.input_dim = Config.INPUT_DIM
        self.conv_filters = Config.CONV_FILTERS
        self.conv_kernel = Config.CONV_KERNEL_SIZE
        # BiGRU output dimension is 2 * hidden_dim
        self.gru_hidden = Config.HIDDEN_DIM
        self.hidden_dim = Config.HIDDEN_DIM * 2
        self.dropout_p = Config.DROPOUT

        # 1. Convolutional Stem
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )
        self.stem_act = nn.GELU()

        # 2. High-Capacity 4-Layer Backbone

        # Block 1
        self.gru_1 = nn.GRU(
            input_size=self.conv_filters,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.affine_1 = AffineStructuralModule(self.hidden_dim)
        self.dropout_1 = nn.Dropout(self.dropout_p)

        # Block 2
        self.gru_2 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.affine_2 = AffineStructuralModule(self.hidden_dim)
        self.dropout_2 = nn.Dropout(self.dropout_p)

        # Block 3
        self.gru_3 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.affine_3 = AffineStructuralModule(self.hidden_dim)
        self.dropout_3 = nn.Dropout(self.dropout_p)

        # Block 4 (Final block, no affine module)
        self.gru_4 = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_4 = nn.Dropout(self.dropout_p)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, Config.NUM_TARGETS)

    def forward(self, inputs, pair_indices, pair_masks):
        """
        Args:
            inputs: (Batch, Seq_Len, 14)
            pair_indices: (Batch, Seq_Len)
            pair_masks: (Batch, Seq_Len)
        """
        # Stem
        # Permute for Conv1d: (B, C, L)
        x = inputs.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        # Permute back: (B, L, C)
        x = x.permute(0, 2, 1)

        # Block 1
        x, _ = self.gru_1(x)
        x = self.affine_1(x, pair_indices, pair_masks)
        x = self.dropout_1(x)

        # Block 2
        x, _ = self.gru_2(x)
        x = self.affine_2(x, pair_indices, pair_masks)
        x = self.dropout_2(x)

        # Block 3
        x, _ = self.gru_3(x)
        x = self.affine_3(x, pair_indices, pair_masks)
        x = self.dropout_3(x)

        # Block 4 (No Affine Module)
        x, _ = self.gru_4(x)
        x = self.dropout_4(x)

        # Head
        out = self.head(x)

        return out
