import torch
import torch.nn as nn
from library.config import Config


class DecoupledInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module.

    Implements the structural injection mechanism with:
    1. Point-to-Point Gathering of neighbor states.
    2. Zero-Masking for unpaired bases (no self-loops).
    3. Decoupled Message derivation (neighbor content only).
    4. Context-Aware Gating.
    5. Post-Normalization (LayerNorm after residual).
    """

    def __init__(self, hidden_dim):
        super(DecoupledInteractionModule, self).__init__()
        # Message pathway: Derived solely from neighbor h_j
        self.W_msg = nn.Linear(hidden_dim, hidden_dim)

        # Gating pathway: Derived from joint context [h_i; h_j]
        self.W_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj_indices, pair_mask):
        """
        Args:
            h (torch.Tensor): Hidden states (Batch, SeqLen, HiddenDim).
            adj_indices (torch.Tensor): Adjacency indices (Batch, SeqLen).
            pair_mask (torch.Tensor): Pair mask (Batch, SeqLen, 1).
        """
        B, L, H = h.shape

        # 1. Gather Neighbor States
        # Expand indices to match hidden dimension: (B, L) -> (B, L, H)
        idx_expanded = adj_indices.unsqueeze(-1).expand(-1, -1, H)
        # Gather: h_neighbor[b, i, :] = h[b, adj_indices[b, i], :]
        h_neighbor = torch.gather(h, 1, idx_expanded)

        # 2. Zero-Masking
        # Force unpaired neighbors to zero vector.
        # pair_mask is 1.0 for paired, 0.0 for unpaired.
        h_neighbor = h_neighbor * pair_mask

        # 3. Decoupled Message
        # m_ij = GELU(W_msg * h_j)
        m = self.act(self.W_msg(h_neighbor))

        # 4. Context-Aware Gating
        # g_ij = Sigmoid(W_gate * [h_i; h_j])
        cat_feat = torch.cat([h, h_neighbor], dim=-1)
        g = self.sigmoid(self.W_gate(cat_feat))

        # 5. Injection
        # h_res = h_i + g_ij * m_ij
        h_res = h + g * m

        # 6. Post-Normalization
        out = self.norm(h_res)

        return out


class DDCGBiGRU(nn.Module):
    """
    Deep Decoupled Channel-Gated BiGRU (DDCG-BiGRU).

    Architecture:
    - Input: (B, 107, 14) One-Hot
    - Stem: Conv1d (K=3, F=256)
    - Backbone: 4 Blocks of [BiGRU -> Interaction -> Dropout]
      (Note: Interaction module is omitted in the final block)
    - Head: Linear -> 5 Targets
    """

    def __init__(self):
        super(DDCGBiGRU, self).__init__()

        # ==========================
        # Convolutional Stem
        # ==========================
        self.stem_conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL,
            padding=Config.CONV_KERNEL // 2,
        )
        self.stem_act = nn.GELU()

        # ==========================
        # Deep Backbone
        # ==========================
        self.num_layers = Config.NUM_LAYERS
        gru_hidden = Config.HIDDEN_SIZE
        bi_hidden = gru_hidden * 2  # Bidirectional output size

        self.gru_blocks = nn.ModuleList()
        self.interaction_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Input dimension: Conv filters for first layer, else previous BiGRU output
            input_dim = Config.CONV_FILTERS if i == 0 else bi_hidden

            # BiGRU Layer
            self.gru_blocks.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=gru_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Decoupled Interaction Module
            # Applied after layers 0, 1, 2. Not applied after the final layer (3).
            if i < self.num_layers - 1:
                self.interaction_blocks.append(DecoupledInteractionModule(bi_hidden))
            else:
                # Placeholder to keep indexing simple, though not used
                self.interaction_blocks.append(nn.Identity())

            self.dropouts.append(nn.Dropout(Config.DROPOUT))

        # ==========================
        # Output Head
        # ==========================
        self.head = nn.Linear(bi_hidden, Config.NUM_TARGETS)

    def forward(self, x, adj_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Input features (B, 107, 14).
            adj_indices (torch.Tensor): Adjacency indices for gathering (B, 107).
            pair_mask (torch.Tensor): Mask for paired bases (B, 107, 1).

        Returns:
            torch.Tensor: Predictions (B, 107, 5).
        """
        # 1. Stem
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = self.stem_act(x)
        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # 2. Backbone
        for i in range(self.num_layers):
            # BiGRU
            # x shape: (B, L, InputDim) -> (B, L, BiHidden)
            x, _ = self.gru_blocks[i](x)

            # Interaction (only for first N-1 layers)
            if i < self.num_layers - 1:
                x = self.interaction_blocks[i](x, adj_indices, pair_mask)

            # Dropout
            x = self.dropouts[i](x)

        # 3. Head
        out = self.head(x)

        return out
