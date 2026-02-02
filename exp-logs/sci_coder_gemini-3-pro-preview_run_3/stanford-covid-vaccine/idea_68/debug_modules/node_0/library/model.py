import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUInteractionModule(nn.Module):
    """
    GLU-Decoupled Structural Injection Module with Deep Stabilized Gate.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Deep Stabilized MLP Gate
        # Input: [h_i; h_j] -> LayerNorm -> GELU -> Linear -> Sigmoid
        self.gate_norm = nn.LayerNorm(2 * hidden_dim)
        self.gate_w1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_w2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h: (B, L, H) Tensor
            pair_indices: (B, L) LongTensor
            pair_masks: (B, L) FloatTensor
        """
        B, L, H = h.shape

        # Gather h_j (Neighbor state)
        # batch_idx: (B, L) where batch_idx[i, :] = i
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)
        h_j = h[batch_idx, pair_indices]  # (B, L, H)

        # Mask h_j: Force to 0 if unpaired
        # pair_masks is (B, L), unsqueeze to (B, L, 1) for broadcasting
        h_j = h_j * pair_masks.unsqueeze(-1)

        # 1. GLU Message Generation
        # If h_j is 0 (unpaired), this becomes bias_c * sigma(bias_g) (Bias-Driven Refinement)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 2. Deep Stabilized Gate Calculation
        cat_input = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)
        z = self.gate_norm(cat_input)
        z = F.gelu(self.gate_w1(z))
        g_ij = torch.sigmoid(self.gate_w2(z))

        # 3. Injection & Normalization
        h_res = h + g_ij * m_ij
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity GLU-Decoupled BiGRU with Deep Stabilized Gating.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Hyperparameters
        input_channels = config.INPUT_CHANNELS
        stem_filters = config.STEM_FILTERS
        hidden_dim = config.HIDDEN_DIM  # Dimension per direction
        total_hidden = config.TOTAL_HIDDEN  # 2 * hidden_dim
        layers = config.LAYERS

        # Convolutional Stem
        self.stem_conv = nn.Conv1d(
            input_channels, stem_filters, kernel_size=3, padding=1
        )
        self.stem_act = nn.GELU()

        # Backbone: 4 Blocks
        self.blocks = nn.ModuleList()
        curr_dim = stem_filters

        for i in range(layers):
            # BiGRU
            gru = nn.GRU(
                input_size=curr_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            curr_dim = total_hidden  # Output of BiGRU is always total_hidden

            # Interaction Module (Layers 0, 1, 2 only; skip final layer)
            interaction = None
            if i < layers - 1:
                interaction = GLUInteractionModule(total_hidden)

            self.blocks.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

        # Output Head
        self.head = nn.Linear(total_hidden, 5)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (B, L, 14)
            pair_indices: (B, L)
            pair_masks: (B, L)
        """
        # x: (B, L, C) -> Conv1d needs (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = x.permute(0, 2, 1)  # (B, L, C)

        h = x
        for block in self.blocks:
            h, _ = block["gru"](h)
            if block["interaction"] is not None:
                h = block["interaction"](h, pair_indices, pair_masks)

        out = self.head(h)
        return out
