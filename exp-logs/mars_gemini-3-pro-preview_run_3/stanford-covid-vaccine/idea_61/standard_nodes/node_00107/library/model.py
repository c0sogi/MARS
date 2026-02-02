import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Bias-Refinement and Stabilization.

    Mechanisms:
    1. Gather: Retrieves neighbor features h_j based on pair_indices.
    2. Masking: Forces h_j=0 for unpaired bases.
    3. Bias-Refinement: For unpaired bases, the message becomes GELU(bias),
       acting as a learnable loop embedding.
    4. Stabilization: Uses Internal Gate Normalization and Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message Projection: m_ij = GELU(W_msg * h_j + b_msg)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating Mechanism: g_ij = Sigmoid(W_g2 * GELU(LN(W_g1 * [h_i; h_j])))
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Gate Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization for residual connection
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor (Batch, Seq, HiddenDim)
            pair_indices: LongTensor (Batch, Seq) - Indices of paired bases.
            pair_mask: Tensor (Batch, Seq) - 1.0 if paired, 0.0 if unpaired.
        """
        B, L, D = x.shape

        # 1. Gather Neighbor Features (h_j)
        # Expand indices to match feature dimension: (B, L, D)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        # Gather: h_j[b, i, :] = x[b, pair_indices[b, i], :]
        h_j = torch.gather(x, 1, idx)

        # 2. Mask Unpaired Neighbors
        # If unpaired, h_j becomes 0 vector.
        # pair_mask is (B, L) -> (B, L, 1)
        mask = pair_mask.unsqueeze(-1)
        h_j = h_j * mask

        # 3. Compute Message
        # If h_j is 0 (unpaired), this becomes GELU(bias), learning a loop embedding.
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Compute Gate
        # Concatenate current node h_i and neighbor h_j
        cat_input = torch.cat([x, h_j], dim=-1)

        # Stabilized MLP Gate
        z_raw = self.gate_proj1(cat_input)
        z_norm = self.gate_norm(z_raw)  # Internal Normalization
        z_act = F.gelu(z_norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)  # No logit normalization to allow saturation

        # 5. Injection & Residual
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Stabilized Decoupled Bias-Refined BiGRU.

    Architecture:
    - Conv1d Stem
    - 4 Layers of BiGRU (Hidden=384*2=768)
    - Structural Interaction Modules interleaved (after layers 0, 1, 2)
    - Linear Head
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Convolutional Stem
        # ==========================================
        self.conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            padding=Config.CONV_KERNEL_SIZE // 2,
        )

        # ==========================================
        # 2. Backbone Construction
        # ==========================================
        self.num_layers = Config.NUM_LAYERS
        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # BiGRU Output Dimension = Hidden * 2
        gru_hidden_dim = Config.HIDDEN_DIM
        backbone_dim = gru_hidden_dim * 2

        current_input_dim = Config.CONV_FILTERS

        for i in range(self.num_layers):
            # BiGRU Layer
            self.grus.append(
                nn.GRU(
                    input_size=current_input_dim,
                    hidden_size=gru_hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Update input dimension for next layer
            current_input_dim = backbone_dim

            # Interaction Module (except after the final block)
            if i < self.num_layers - 1:
                self.interactions.append(InteractionModule(backbone_dim))
            else:
                self.interactions.append(None)  # Placeholder

            # Dropout
            self.dropouts.append(nn.Dropout(Config.DROPOUT))

        # ==========================================
        # 3. Output Head
        # ==========================================
        self.head = nn.Linear(backbone_dim, Config.OUTPUT_DIM)

    def forward(self, inputs, pair_index=None, pair_mask=None, **kwargs):
        """
        Args:
            inputs: (Batch, SeqLen, InputDim)
            pair_index: (Batch, SeqLen)
            pair_mask: (Batch, SeqLen)
        """
        # 1. Stem
        # Permute to (B, C, L) for Conv1d
        x = inputs.transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x)
        # Permute back to (B, L, C) for RNN
        x = x.transpose(1, 2)

        # 2. Backbone
        for i in range(self.num_layers):
            # BiGRU
            x, _ = self.grus[i](x)

            # Interaction (if exists for this layer)
            interaction_module = self.interactions[i]
            if interaction_module is not None:
                # We need pair_index and pair_mask for interaction
                if pair_index is not None and pair_mask is not None:
                    x = interaction_module(x, pair_index, pair_mask)

            # Dropout
            x = self.dropouts[i](x)

        # 3. Head
        out = self.head(x)

        return out
