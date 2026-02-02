import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block for the Dense Backbone.
    Follows the structure: ReLU -> Dilated Conv1d -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super(DenseBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=Config.KERNEL_SIZE,
                dilation=dilation,
                padding=dilation,  # Maintains sequence length
                bias=True,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ProjectedInteractionLayer(nn.Module):
    """
    High-Fidelity Linear Interaction Layer.
    Projects dense history to a latent space, gathers partner features,
    and fuses them via concatenation.
    """

    def __init__(self, in_channels, latent_dim):
        super(ProjectedInteractionLayer, self).__init__()
        # Linear projection (1x1 conv) to latent dimension
        # No non-linearity here to avoid information bottleneck distortion
        self.projection = nn.Conv1d(in_channels, latent_dim, kernel_size=1)

    def forward(self, x, partner_indices, partner_mask):
        """
        Args:
            x: (Batch, In_Channels, Seq_Len) - The dense history tensor
            partner_indices: (Batch, Seq_Len) - Indices of paired bases
            partner_mask: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        # 1. Project to latent space
        # Shape: (Batch, Latent_Dim, Seq_Len)
        h = self.projection(x)

        # 2. Gather Partner Features
        batch_size, channels, seq_len = h.shape

        # Expand indices for gather: (Batch, 1, Seq_Len) -> (Batch, Channels, Seq_Len)
        idx_expanded = partner_indices.unsqueeze(1).expand(-1, channels, -1)

        # Gather: h_partner[b, c, i] = h[b, c, partner_indices[b, i]]
        # Note: partner_indices contains dummy 0s for unpaired bases, but
        # the mask will zero these out in the next step.
        h_partner = torch.gather(h, 2, idx_expanded)

        # 3. Apply Null-Mask
        # partner_mask is (Batch, Seq_Len). Expand to (Batch, Channels, Seq_Len)
        mask_expanded = partner_mask.unsqueeze(1)
        h_partner = h_partner * mask_expanded

        # 4. Concatenate Local and Partner Vectors
        # Shape: (Batch, 2 * Latent_Dim, Seq_Len)
        out = torch.cat([h, h_partner], dim=1)

        return out


class RNANet(nn.Module):
    """
    Projected Latent-Interaction Dense Network.
    Combines a Dense Dilated TCN backbone with a symmetric linear interaction mechanism.
    """

    def __init__(self):
        super(RNANet, self).__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_DIM
        growth_rate = Config.GROWTH_RATE
        dilations = Config.DILATIONS
        dropout = Config.DROPOUT
        latent_dim = Config.LATENT_DIM
        rnn_hidden = Config.RNN_HIDDEN_DIM
        num_targets = Config.NUM_TARGETS
        rnn_layers = Config.RNN_LAYERS

        # --- 1. Stem ---
        # Initial convolution to map input features to growth_rate width
        self.stem = nn.Conv1d(input_dim, growth_rate, kernel_size=3, padding=1)

        # --- 2. Dense Backbone ---
        self.blocks = nn.ModuleList()
        current_channels = growth_rate

        for d in dilations:
            # Each block takes the concatenation of all previous outputs
            block = DenseBlock(
                current_channels, growth_rate, dilation=d, dropout=dropout
            )
            self.blocks.append(block)
            current_channels += growth_rate

        self.backbone_out_channels = current_channels

        # --- 3. Interaction Layer ---
        self.interaction = ProjectedInteractionLayer(
            self.backbone_out_channels, latent_dim
        )

        # Output dimension after concatenation (Latent + Partner_Latent)
        interaction_out_dim = latent_dim * 2

        # --- 4. Global Aggregator (BiGRU) ---
        self.gru = nn.GRU(
            input_size=interaction_out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        # BiGRU output is hidden_size * 2
        gru_out_dim = rnn_hidden * 2

        # --- 5. Output Head ---
        self.head = nn.Linear(gru_out_dim, num_targets)

    def forward(self, x, partner_indices, partner_mask, targets=None):
        """
        Args:
            x: (Batch, Seq_Len, Input_Dim)
            partner_indices: (Batch, Seq_Len)
            partner_mask: (Batch, Seq_Len)
            targets: Unused, kept for compatibility with training loop signature
        """
        # Permute to (Batch, Channels, Seq_Len) for Conv1d operations
        x = x.permute(0, 2, 1)

        # 1. Stem
        out = self.stem(x)

        # 2. Dense Backbone
        # Maintain a list of feature tensors to concatenate
        features = [out]

        for block in self.blocks:
            # Concatenate all previous features (Dense Connection)
            in_feat = torch.cat(features, dim=1)
            # Compute new block output
            new_feat = block(in_feat)
            # Add to history
            features.append(new_feat)

        # Final dense representation: Concatenate all history
        # Shape: (Batch, Backbone_Out_Channels, Seq_Len)
        h_dense = torch.cat(features, dim=1)

        # 3. Interaction Layer
        # Projects, gathers, and fuses
        # Shape: (Batch, 256, Seq_Len)
        h_interact = self.interaction(h_dense, partner_indices, partner_mask)

        # 4. Global Aggregation (BiGRU)
        # Permute back to (Batch, Seq_Len, Channels) for RNN
        h_rnn_in = h_interact.permute(0, 2, 1)

        self.gru.flatten_parameters()
        h_rnn_out, _ = self.gru(h_rnn_in)

        # 5. Output Head
        logits = self.head(h_rnn_out)

        return logits
