import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class EnhancedGatedBlock(nn.Module):
    """
    Implements the Stabilized Decoupled Interaction Module with Enhanced-Context Gating.

    Key Innovations:
    1. Point-to-Point Gather with Zero-Masking for unpaired bases.
    2. Bias-Refined GLU Message mechanism.
    3. Enhanced-Context Gating using [h_i; h_j; h_i * h_j].
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # =====================================================================
        # GLU Message Components
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # =====================================================================
        self.w_c = nn.Linear(hidden_dim, hidden_dim)
        self.w_g = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Enhanced-Context Gate Components
        # Input: [h_i; h_j; h_i * h_j] -> 3 * hidden_dim
        # =====================================================================
        self.gate_in_dim = 3 * hidden_dim

        # Wide Projection: Projects to full width (768)
        self.gate_mlp_in = nn.Linear(self.gate_in_dim, hidden_dim)

        # Internal Normalization & Activation
        self.gate_ln = nn.LayerNorm(hidden_dim)

        # Output Projection (No Logit Norm, allows saturation)
        self.gate_mlp_out = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # Post-Normalization
        # =====================================================================
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq, Hidden)
            pair_indices: Tensor of shape (Batch, Seq) containing indices of paired bases.
                          Unpaired bases are denoted by -1.
        """
        batch_size, seq_len, _ = x.shape

        # ---------------------------------------------------------------------
        # 1. Gather h_j (Point-to-Point)
        # ---------------------------------------------------------------------
        # Create mask for paired bases: 1 if paired, 0 if unpaired (-1)
        mask = (pair_indices != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Prepare indices for gather. Replace -1 with 0 to prevent index errors.
        # The result at these positions will be masked out anyway.
        gather_indices = pair_indices.clone()
        gather_indices[pair_indices == -1] = 0

        # Expand indices to match hidden dimension: (B, L, H)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )

        # Gather h_j: Output[b, i, k] = x[b, gather_indices[b, i], k]
        h_j = torch.gather(x, 1, gather_indices_expanded)

        # Apply Zero-Masking: Force h_j = 0 for unpaired bases
        h_j = h_j * mask

        # ---------------------------------------------------------------------
        # 2. GLU Message (Bias-Refined)
        # ---------------------------------------------------------------------
        # For unpaired bases (h_j=0), this becomes b_c * sigma(b_g) (Loop Embedding)
        msg_content = self.w_c(h_j)
        msg_gate = torch.sigmoid(self.w_g(h_j))
        m_ij = msg_content * msg_gate

        # ---------------------------------------------------------------------
        # 3. Enhanced-Context Gate
        # ---------------------------------------------------------------------
        # Explicitly compute element-wise product for feature matching
        h_prod = x * h_j

        # Concatenate: [h_i; h_j; h_i * h_j]
        cat_input = torch.cat([x, h_j, h_prod], dim=-1)

        # Wide MLP Projection
        z_raw = self.gate_mlp_in(cat_input)
        z_norm = self.gate_ln(z_raw)
        z_act = F.gelu(z_norm)

        # Gate calculation
        g_ij = torch.sigmoid(self.gate_mlp_out(z_act))

        # ---------------------------------------------------------------------
        # 4. Injection & Post-Normalization
        # ---------------------------------------------------------------------
        # Additive injection
        h_res = x + g_ij * m_ij

        # Stabilize stack
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Enhanced-Context Synthesis Model.

    Structure:
    1. 1D Convolutional Stem
    2. 4-Layer Backbone (BiGRU + EnhancedGatedBlock)
    3. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.hidden_dim = Config.HIDDEN_DIM * 2  # BiGRU (384*2 = 768)
        input_dim = Config.INPUT_DIM  # 14
        conv_filters = Config.CONV_FILTERS  # 256
        conv_kernel = Config.CONV_KERNEL  # 3
        num_layers = Config.NUM_LAYERS  # 4
        dropout = Config.DROPOUT  # 0.1

        # ---------------------------------------------------------------------
        # 1. Convolutional Stem
        # ---------------------------------------------------------------------
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_filters,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
        )
        self.conv_act = nn.GELU()

        # ---------------------------------------------------------------------
        # 2. Deep High-Capacity Backbone
        # ---------------------------------------------------------------------
        self.blocks = nn.ModuleList()

        # First layer input comes from Conv (256), subsequent from previous block (768)
        gru_input_dim = conv_filters

        for i in range(num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=Config.HIDDEN_DIM,  # 384
                batch_first=True,
                bidirectional=True,
            )

            # Interaction Module (EnhancedGatedBlock)
            # Operates on the full BiGRU output width (768)
            interaction = EnhancedGatedBlock(self.hidden_dim)

            self.blocks.append(nn.ModuleList([gru, interaction]))

            # Update input dim for next layer
            gru_input_dim = self.hidden_dim

        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 3. Output Head
        # ---------------------------------------------------------------------
        self.head = nn.Linear(self.hidden_dim, 5)

    def forward(self, inputs, pair_indices):
        """
        Args:
            inputs: (Batch, Seq, 14)
            pair_indices: (Batch, Seq)
        Returns:
            logits: (Batch, Seq, 5)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.permute(0, 2, 1)

        # Stem
        x = self.conv(x)
        x = self.conv_act(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone Processing
        for gru, interaction in self.blocks:
            # BiGRU Processing
            # gru_out: (Batch, Seq, 2*Hidden)
            gru_out, _ = gru(x)

            # Apply Dropout
            gru_out = self.dropout(gru_out)

            # Interaction Processing (Decoupled & Gated)
            # x is updated to the output of the interaction block
            x = interaction(gru_out, pair_indices)

        # Head Projection
        logits = self.head(x)

        return logits
