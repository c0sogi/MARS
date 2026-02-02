import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Implements the Decoupled Structural Interaction with Unmasked Bias Propagation.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message Projection: W_msg * h_j + b_msg
        # We use a Linear layer. For unpaired bases (input 0), this outputs the bias.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Channel-Wise Gating Projection: W_gate * [h_i; h_j]
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bpp_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            bpp_indices: Tensor of shape (Batch, Seq_Len) containing indices of paired bases (-1 for unpaired).
        """
        B, L, C = x.shape

        # 1. Create Masks and Safe Indices
        # Mask is 1.0 if paired, 0.0 if unpaired
        mask = (bpp_indices != -1).unsqueeze(-1).float().to(x.device)  # (B, L, 1)

        # Replace -1 with 0 to ensure valid gathering indices.
        # The value gathered at index 0 will be masked out later if the original index was -1.
        safe_indices = bpp_indices.clone()
        safe_indices[bpp_indices == -1] = 0

        # 2. Gather Neighbor States
        # Calculate flat indices to gather from the (B*L, C) flattened tensor
        batch_offsets = (torch.arange(B, device=x.device) * L).unsqueeze(1)  # (B, 1)
        flat_indices = (safe_indices + batch_offsets).view(-1)  # (B*L)

        flat_x = x.reshape(-1, C)
        h_neighbor = torch.index_select(flat_x, 0, flat_indices)
        h_neighbor = h_neighbor.view(B, L, C)

        # 3. Zero-Masking (Input Side)
        # Explicitly force the gathered vector to 0 for unpaired bases.
        # This prevents information leakage from index 0 and ensures "Bias Propagation" relies on zero-input.
        h_neighbor = h_neighbor * mask

        # 4. Decoupled Message Calculation
        # m_ij = GELU(W * h_j + b)
        # For unpaired bases, h_neighbor is 0, so m_ij = GELU(bias).
        # This constant bias vector acts as a learnable "unpaired" structural embedding.
        m_ij = F.gelu(self.msg_proj(h_neighbor))

        # 5. Channel-Wise Gating
        # g_ij = Sigmoid(W * [h_i; h_j])
        # Allows the current state h_i to control how much of the structural message (or bias) to accept.
        cat_input = torch.cat([x, h_neighbor], dim=-1)
        g_ij = torch.sigmoid(self.gate_proj(cat_input))

        # 6. Injection and Residual Update
        # h_res = h_i + g_ij * m_ij
        update = g_ij * m_ij
        x_res = x + self.dropout(update)

        # 7. Post-Normalization
        # Stabilizes the deep backbone
        x_out = self.norm(x_res)

        return x_out


class RNAModel(nn.Module):
    """
    Deep Decoupled Channel-Gated BiGRU with Unmasked Bias Propagation.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.hidden_dim = config.HIDDEN_DIM  # 384
        self.num_layers = config.NUM_LAYERS  # 4
        self.input_channels = config.INPUT_CHANNELS  # 14
        self.output_channels = config.OUTPUT_CHANNELS  # 5
        self.dropout_p = config.DROPOUT

        # --- 1. Convolutional Stem ---
        # Projects sparse one-hot inputs into dense embedding space.
        # Aggregates local context via kernel size 3.
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=256,
            kernel_size=config.KERNEL_SIZE,
            padding=config.KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()

        # --- 2. Deep Stabilized Backbone ---
        self.gru_layers = nn.ModuleList()
        self.interaction_modules = nn.ModuleList()

        # Dimensions logic
        current_dim = 256  # Output of Stem
        gru_hidden = self.hidden_dim
        bidirectional = True
        gru_out_dim = gru_hidden * 2 if bidirectional else gru_hidden  # 768

        for i in range(self.num_layers):
            # BiGRU Layer
            self.gru_layers.append(
                nn.GRU(
                    input_size=current_dim,
                    hidden_size=gru_hidden,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )

            # Structural Interaction Module
            # Interleaved after every block EXCEPT the final one.
            if i < self.num_layers - 1:
                self.interaction_modules.append(
                    StructuralInteractionModule(gru_out_dim, dropout=self.dropout_p)
                )

            # Next layer input is the output of the current BiGRU (768)
            current_dim = gru_out_dim

        self.dropout = nn.Dropout(self.dropout_p)

        # --- 3. Output Head ---
        self.head = nn.Linear(gru_out_dim, self.output_channels)

    def forward(self, inputs, bpp_indices):
        """
        Args:
            inputs: (Batch, Seq_Len, 14)
            bpp_indices: (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = inputs.permute(0, 2, 1)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back for RNN: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone Processing
        for i in range(self.num_layers):
            # BiGRU
            # Returns (output, h_n). We use output.
            x, _ = self.gru_layers[i](x)

            # Regularization
            x = self.dropout(x)

            # Interaction (if present for this layer)
            if i < len(self.interaction_modules):
                x = self.interaction_modules[i](x, bpp_indices)

        # Head
        logits = self.head(x)

        return logits
