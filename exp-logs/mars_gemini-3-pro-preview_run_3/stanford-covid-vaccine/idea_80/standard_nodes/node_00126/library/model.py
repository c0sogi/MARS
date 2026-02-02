import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class UnifiedGLUInteraction(nn.Module):
    """
    Implements the Unified GLU-Decoupled Interaction Module.

    Features:
    - Point-to-Point Gather: Retrieves features from paired bases.
    - Input Zero-Masking: Forces unpaired base interactions to zero, relying on bias terms.
    - Bias-Refined GLU: Uses Gated Linear Units for expressive messaging.
    - Full-Rank Stabilized Gate: Controls the injection of structural information.
    - Decoupled Path: Does not concatenate source features in the message generation, only in gating.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message components: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        self.w_c = nn.Linear(hidden_dim, hidden_dim)
        self.w_g = nn.Linear(hidden_dim, hidden_dim)

        # Gating components: g_ij = sigmoid(W_out * GELU(LayerNorm(W_in * [h_i; h_j])))
        # Input is concatenation of source and target features -> 2 * hidden_dim
        self.w_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm_gate = nn.LayerNorm(hidden_dim)
        self.w_out = nn.Linear(hidden_dim, hidden_dim)

        # Output Normalization
        self.layer_norm_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, bpp_indices, bpp_masks):
        """
        Args:
            h: Hidden states (Batch, Seq, Hidden)
            bpp_indices: Indices of paired bases (Batch, Seq)
            bpp_masks: Mask indicating paired status (1.0 paired, 0.0 unpaired) (Batch, Seq)
        """
        batch_size, seq_len, _ = h.shape

        # 1. Gather h_j (Target features)
        # Create batch indices grid
        batch_idx = (
            torch.arange(batch_size, device=h.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Gather features: h[b, bpp_indices[b, i], :]
        h_j = h[batch_idx, bpp_indices]  # (Batch, Seq, Hidden)

        # 2. Input Zero-Masking
        # If unpaired (mask=0), force h_j to 0.
        # The linear layers will then output their bias terms (b_c, b_g).
        mask_expanded = bpp_masks.unsqueeze(-1)  # (Batch, Seq, 1)
        h_j = h_j * mask_expanded

        # 3. GLU Message (Bias-Refined)
        # m = Linear(h_j) * Sigmoid(Linear(h_j))
        m_ij = self.w_c(h_j) * torch.sigmoid(self.w_g(h_j))

        # 4. Full-Rank Stabilized Gate
        # Concatenate h_i (current) and h_j (neighbor/zero)
        cat_input = torch.cat([h, h_j], dim=-1)  # (Batch, Seq, 2*Hidden)

        # Wide projection -> Norm -> Act -> Projection -> Sigmoid
        z_raw = self.w_in(cat_input)
        z_norm = self.layer_norm_gate(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.w_out(z_act))

        # 5. Injection with Residual Connection
        # h_struct = h + dropout(g * m)
        update = g_ij * m_ij
        update = self.dropout(update)
        h_struct = h + update

        # 6. Post-Normalization
        h_out = self.layer_norm_out(h_struct)

        return h_out


class ResidualBiGRUBlock(nn.Module):
    """
    Implements a Bidirectional GRU with a Vertical Residual Connection.

    Logic:
    - If dimensions match (Layers 2-4): Output = Input + BiGRU(Input)
    - If dimensions mismatch (Layer 1): Output = BiGRU(Input)
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1, is_residual=True):
        super().__init__()
        self.is_residual = is_residual
        # BiGRU output dimension is 2 * hidden_dim
        self.output_dim = hidden_dim * 2

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        out, _ = self.gru(x)  # out: (Batch, Seq, 2*Hidden)
        out = self.dropout(out)

        if self.is_residual and x.shape[-1] == out.shape[-1]:
            return x + out
        else:
            return out


class RNAModel(nn.Module):
    """
    Deep Residual High-Capacity BiGRU with Unified GLU-Refinement.

    Architecture:
    1. Convolutional Stem (1D Conv)
    2. 4 Layers of [Residual BiGRU -> Unified GLU Interaction]
    3. Linear Output Head
    """

    def __init__(self, config: Config):
        super().__init__()

        # Configuration
        self.input_dim = config.input_dim  # 14
        self.cnn_filters = config.cnn_filters  # 256
        self.gru_hidden = config.hidden_dim  # 384 (per direction)
        self.backbone_dim = self.gru_hidden * 2  # 768 (total backbone width)
        self.num_layers = config.num_layers  # 4
        self.dropout_rate = config.dropout  # 0.1
        self.num_targets = config.num_targets  # 5
        self.kernel_size = config.kernel_size  # 3

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs to dense embeddings and aggregates local context
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.cnn_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Residual Backbone
        self.layers = nn.ModuleList()

        # Layer 1: Adapts dimension from cnn_filters (256) to backbone_dim (768)
        # Not strictly residual in the RNN part due to dim change
        self.layers.append(
            nn.ModuleDict(
                {
                    "gru": ResidualBiGRUBlock(
                        self.cnn_filters,
                        self.gru_hidden,
                        self.dropout_rate,
                        is_residual=False,
                    ),
                    "interact": UnifiedGLUInteraction(
                        self.backbone_dim, self.dropout_rate
                    ),
                }
            )
        )

        # Layers 2 to N: Fully Residual Blocks
        # Input (768) -> Residual BiGRU -> Residual Interaction -> Output (768)
        for _ in range(1, self.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "gru": ResidualBiGRUBlock(
                            self.backbone_dim,
                            self.gru_hidden,
                            self.dropout_rate,
                            is_residual=True,
                        ),
                        "interact": UnifiedGLUInteraction(
                            self.backbone_dim, self.dropout_rate
                        ),
                    }
                )
            )

        # 3. Output Head
        self.head = nn.Linear(self.backbone_dim, self.num_targets)

    def forward(self, x, bpp_indices, bpp_masks):
        """
        Forward pass of the model.

        Args:
            x: Input features (Batch, Seq, 14)
            bpp_indices: BPP partner indices (Batch, Seq)
            bpp_masks: BPP paired masks (Batch, Seq)

        Returns:
            logits: Predicted values (Batch, Seq, 5)
        """
        # Permute for Conv1d: (Batch, Channels, Seq)
        x = x.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)

        # Permute back: (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)

        # Apply Backbone Layers
        for layer in self.layers:
            # 1. RNN Block (Temporal Context)
            x = layer["gru"](x)

            # 2. Interaction Block (Structural Context)
            x = layer["interact"](x, bpp_indices, bpp_masks)

        # Apply Head
        logits = self.head(x)

        return logits
