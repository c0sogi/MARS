import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputAwareGatingUnit(nn.Module):
    """
    Implements the Input-Aware Channel-Gating mechanism.
    Injects static stem features into the gating logic to verify physical base-pairing
    compatibility at deep layers.
    """

    def __init__(self, d_model, d_stem):
        super().__init__()
        self.msg_proj = nn.Linear(d_model, d_model)
        # Gate input: [h_i; h_j; x_stem_i; x_stem_j]
        self.gate_proj = nn.Linear(2 * d_model + 2 * d_stem, d_model)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, x_stem, pair_indices, pair_masks):
        """
        Args:
            h: (B, L, d_model) - Current hidden states from BiGRU
            x_stem: (B, L, d_stem) - Static stem embeddings
            pair_indices: (B, L) - Indices of paired bases
            pair_masks: (B, L, 1) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = h.shape
        D_stem = x_stem.shape[-1]

        # Expand indices for gathering to match feature dimensions
        # pair_indices is (B, L) -> (B, L, D) and (B, L, D_stem)
        idx_h = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        idx_stem = pair_indices.unsqueeze(-1).expand(-1, -1, D_stem)

        # Gather neighbor states (h_j) and neighbor stem features (x_stem_j)
        # torch.gather retrieves the rows specified by pair_indices along the sequence dimension (dim=1)
        h_j = torch.gather(h, 1, idx_h)
        x_stem_j = torch.gather(x_stem, 1, idx_stem)

        # Apply Zero-Masking: Force gathered vectors to zero if the position is unpaired.
        # This removes self-loops (where pair_index[i] == i) and invalid connections.
        h_j = h_j * pair_masks
        x_stem_j = x_stem_j * pair_masks

        # Compute Non-Linear Message
        m = self.act(self.msg_proj(h_j))

        # Compute Input-Aware Gate
        # Concatenate current state, neighbor state, current stem, neighbor stem
        gate_input = torch.cat([h, h_j, x_stem, x_stem_j], dim=-1)
        g = self.sigmoid(self.gate_proj(gate_input))

        # Channel-Wise Gated Residual Update
        # h_res = h_i + g_ij * m_ij
        out = h + g * m
        return out


class DeepInputAwareBiGRU(nn.Module):
    """
    Main architecture: 4-Layer Bidirectional GRU with Interleaved Input-Aware Structural Injection.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.input_channels = Config.INPUT_CHANNELS
        self.stem_channels = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_p = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embedding and aggregates local k-mers
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.stem_channels,
            kernel_size=Config.CONV_KERNEL_SIZE,
            padding=Config.CONV_KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()

        # Projection to Backbone Dimension
        # Maps the stem output (256) to the BiGRU hidden dimension (384)
        self.embedding_proj = nn.Linear(self.stem_channels, self.hidden_dim)

        # 2. Deep Refinement Backbone
        self.gru_blocks = nn.ModuleList()
        self.interaction_blocks = nn.ModuleList()
        self.norm_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # BiGRU hidden_size is half of hidden_dim because it's bidirectional (concat output)
        gru_hidden = self.hidden_dim // 2

        for i in range(self.num_layers):
            # BiGRU Layer
            self.gru_blocks.append(
                nn.GRU(
                    input_size=self.hidden_dim,
                    hidden_size=gru_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Input-Aware Interaction Module
            # Applied in all blocks except the final one
            if i < self.num_layers - 1:
                self.interaction_blocks.append(
                    InputAwareGatingUnit(self.hidden_dim, self.stem_channels)
                )
            else:
                self.interaction_blocks.append(None)

            # Layer Normalization (Post-Norm)
            self.norm_blocks.append(nn.LayerNorm(self.hidden_dim))

            # Dropout
            self.dropouts.append(nn.Dropout(self.dropout_p))

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, features, pair_indices, pair_masks):
        """
        Args:
            features: (B, L, 14)
            pair_indices: (B, L)
            pair_masks: (B, L, 1)
        """
        # Permute for Conv1d: (B, 14, L)
        x = features.permute(0, 2, 1)

        # Stem Processing
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back to sequence format: (B, L, 256)
        # Retain x_stem for use in interaction modules
        x_stem = x.permute(0, 2, 1)

        # Project to backbone dimension
        x = self.embedding_proj(x_stem)

        # Backbone Processing
        for i in range(self.num_layers):
            # BiGRU
            # gru_out: (B, L, 2 * hidden_size) -> (B, L, 384)
            gru_out, _ = self.gru_blocks[i](x)

            # Interaction
            if self.interaction_blocks[i] is not None:
                # Apply Input-Aware Gating
                # Injects structural info and updates gru_out residually
                x_inter = self.interaction_blocks[i](
                    gru_out, x_stem, pair_indices, pair_masks
                )
                # Post-Norm
                x = self.norm_blocks[i](x_inter)
            else:
                # Final block: No interaction, just Norm
                x = self.norm_blocks[i](gru_out)

            # Dropout
            x = self.dropouts[i](x)

        # Output Head
        out = self.head(x)
        return out
