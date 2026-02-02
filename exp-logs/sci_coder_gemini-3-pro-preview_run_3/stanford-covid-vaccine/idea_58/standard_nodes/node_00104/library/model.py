import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StabilizedInteractionModule(nn.Module):
    """
    Implements the Stabilized Decoupled Structural Injection module.

    Key features:
    1. Point-to-Point Gather based on structure indices.
    2. Input Zero-Masking for unpaired bases.
    3. Bias-Refined Message passing (GELU(Wh + b)).
    4. Internal Gate Normalization to ensure stability.
    5. Post-Normalization of the residual stream.
    """

    def __init__(self, hidden_dim):
        super(StabilizedInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message generation: W_msg * h_j + b_msg
        # For unpaired bases (h_j=0), this learns a bias embedding.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gate generation MLP
        # Step 1: Project Joint Context [h_i; h_j] -> z_raw
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

        # Step 2: Internal Normalization (Stabilization)
        self.gate_norm = nn.LayerNorm(hidden_dim)

        # Step 3: Logit Projection -> logits
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Post-Normalization for the residual block
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Args:
            x: Tensor (Batch, Seq_Len, Hidden_Dim) - The hidden states h_i
            bpp_indices: Tensor (Batch, Seq_Len) - Indices of paired bases j
            bpp_mask: Tensor (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, _ = x.shape

        # 1. Gather h_j
        # Expand indices to match hidden dim: (B, L, H)
        gather_indices = bpp_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        # Gather along the sequence dimension (dim=1)
        h_j = torch.gather(x, 1, gather_indices)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0.
        mask = bpp_mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. Decoupled Message (Bias-Refined)
        # m_ij = GELU(W_msg * h_j + b_msg)
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate h_i (x) and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2*H)

        # Project
        z_raw = self.gate_proj1(cat_input)

        # Internal Norm (Crucial for deep network stability)
        z_norm = self.gate_norm(z_raw)

        # Activation
        z_act = F.gelu(z_norm)

        # Logits
        logits = self.gate_proj2(z_act)

        # Sigmoid (No logit norm, allowing saturation)
        g_ij = torch.sigmoid(logits)

        # 5. Injection (Residual)
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HCSDBiGRU(nn.Module):
    """
    High-Capacity Stabilized Decoupled BiGRU (HCSD-BiGRU)

    Structure:
    - Conv1d Stem
    - 4 Blocks of BiGRU (384 hidden dim per direction)
    - Interleaved Stabilized Interaction Modules (after layers 1, 2, 3)
    - Linear Head
    """

    def __init__(self):
        super(HCSDBiGRU, self).__init__()

        # Hyperparameters from Config
        self.input_channels = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.conv_kernel = Config.CONV_KERNEL
        self.hidden_dim = Config.HIDDEN_DIM  # 384 per direction
        self.num_layers = Config.NUM_LAYERS  # 4
        self.dropout_prob = Config.DROPOUT
        self.num_targets = 5  # Multi-task learning on all 5 columns

        # BiGRU outputs 2 * hidden_dim
        total_hidden_dim = self.hidden_dim * 2  # 768

        # 1. Convolutional Stem
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )
        self.stem_act = nn.GELU()

        # 2. Backbone Layers
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # First layer takes conv output, others take previous GRU/Interaction output
            input_dim = self.conv_filters if i == 0 else total_hidden_dim

            # BiGRU Layer
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)
            self.dropouts.append(nn.Dropout(self.dropout_prob))

            # Interaction Module
            # Applied after every block EXCEPT the final one
            if i < self.num_layers - 1:
                interaction = StabilizedInteractionModule(total_hidden_dim)
                self.interaction_layers.append(interaction)

        # 3. Output Head
        self.head = nn.Linear(total_hidden_dim, self.num_targets)

    def forward(self, features, bpp_indices, bpp_mask):
        """
        Args:
            features: (Batch, Seq_Len, 14)
            bpp_indices: (Batch, Seq_Len)
            bpp_mask: (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, C, L)
        x = features.permute(0, 2, 1)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back for GRU: (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone Blocks
        for i in range(self.num_layers):
            # GRU
            x, _ = self.gru_layers[i](x)
            x = self.dropouts[i](x)

            # Interaction (if configured for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, bpp_indices, bpp_mask)

        # Head
        out = self.head(x)

        return out
