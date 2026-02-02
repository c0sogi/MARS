import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block used within the Dense TCN backbone.
    It consists of Conv1d -> BatchNorm -> ReLU -> Dropout.
    The dense connectivity logic (concatenation of inputs) is handled in the main network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DenseDilatedBlock, self).__init__()
        # Calculate padding to maintain sequence length: P = (D * (K-1)) / 2
        # We assume kernel_size is odd (e.g., 3) so padding is an integer.
        self.padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class LatentPartnerGather(nn.Module):
    """
    Module that compresses features and gathers partner features based on structure indices.
    This allows the model to incorporate the latent context of paired bases.
    """

    def __init__(self, in_channels, latent_dim):
        super(LatentPartnerGather, self).__init__()
        # 1x1 Convolution to compress high-dimensional dense features
        self.projector = nn.Conv1d(in_channels, latent_dim, kernel_size=1)

    def forward(self, x, partner_indices):
        """
        Args:
            x: Tensor of shape (Batch, Channels, Seq_Len)
            partner_indices: Tensor of shape (Batch, Seq_Len) containing indices of paired bases.
        Returns:
            Tensor of shape (Batch, Latent_Dim * 2, Seq_Len)
        """
        # 1. Compress dimensionality
        z = self.projector(x)  # (B, Latent, L)

        # 2. Gather Partner Features
        B, C, L = z.shape

        # Expand partner_indices to match channel dimension: (B, L) -> (B, C, L)
        # We use unsqueeze and expand to broadcast indices across channels.
        idx = partner_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather features.
        # out[b, c, i] = input[b, c, index[b, c, i]]
        # This pulls the feature vector of the partner base 'j' to position 'i'.
        z_partner = torch.gather(z, 2, idx)

        # 3. Concatenate (Self + Partner)
        out = torch.cat([z, z_partner], dim=1)  # (B, Latent*2, L)

        return out


class DensePartnerAwareNet(nn.Module):
    """
    Dense-Context Partner-Aware Hybrid Network.

    Architecture:
    1. Input Embedding (Stem)
    2. Dense Dilated TCN Backbone (Dense connectivity pattern)
    3. Latent Structural Interaction (Partner Gather)
    4. BiGRU Global Aggregation
    5. Output Head
    """

    def __init__(
        self,
        input_channels=Config.INPUT_CHANNELS,
        tcn_channels=Config.TCN_CHANNELS,
        tcn_layers=Config.TCN_LAYERS,
        kernel_size=Config.TCN_KERNEL_SIZE,
        dropout=Config.DROPOUT,
        latent_dim=Config.LATENT_DIM,
        gru_hidden=Config.GRU_HIDDEN_DIM,
        num_targets=Config.NUM_TARGETS,
    ):
        super(DensePartnerAwareNet, self).__init__()

        # 1. Stem Convolution
        # Maps input features to the TCN channel dimension
        self.stem = nn.Conv1d(input_channels, tcn_channels, kernel_size=1)
        self.stem_bn = nn.BatchNorm1d(tcn_channels)
        self.stem_act = nn.ReLU()

        # 2. Dense Dilated Backbone
        self.blocks = nn.ModuleList()

        # In a DenseNet pattern, the input to layer K is the concatenation of outputs of layers 0..K-1.
        # We treat the Stem output as the initial feature set (Layer 0 equivalent for indexing).
        # Layer 0 Input: Stem (C).
        # Layer 1 Input: Stem + Layer0_Out (2C).
        # Layer k Input: Stem + Layer0_Out + ... + Layer(k-1)_Out ( (k+1)*C ).

        for i in range(tcn_layers):
            dilation = 2**i
            in_ch = tcn_channels * (i + 1)

            block = DenseDilatedBlock(
                in_channels=in_ch,
                out_channels=tcn_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)

        # 3. Latent Structural Interaction
        # Input to this layer is the concatenation of Stem + All Blocks
        # Total channels = tcn_channels * (tcn_layers + 1)
        total_tcn_channels = tcn_channels * (tcn_layers + 1)
        self.interaction = LatentPartnerGather(total_tcn_channels, latent_dim)

        # 4. Global Aggregation (BiGRU)
        # Input is Latent (Self) + Latent (Partner) = latent_dim * 2
        gru_input_dim = latent_dim * 2
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Head
        # BiGRU output is hidden * 2 (for bidirectional)
        self.head = nn.Linear(gru_hidden * 2, num_targets)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Seq_Len, Input_Channels) - Input features including partner identity.
            partner_indices: (Batch, Seq_Len) - Indices mapping each base to its pair.
        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # Permute to (Batch, Channels, Seq_Len) for Conv1d operations
        x = x.permute(0, 2, 1)

        # Stem
        h0 = self.stem(x)
        h0 = self.stem_bn(h0)
        h0 = self.stem_act(h0)

        # Dense Backbone
        # Keep track of all feature maps for dense concatenation
        features = [h0]

        for block in self.blocks:
            # Concatenate all previous features along channel dim
            curr_in = torch.cat(features, dim=1)
            out = block(curr_in)
            features.append(out)

        # Concatenate everything for the interaction layer
        total_features = torch.cat(features, dim=1)  # (B, Total_C, L)

        # Latent Interaction (Partner Gathering)
        # Returns (B, Latent*2, L)
        struct_features = self.interaction(total_features, partner_indices)

        # Permute back to (Batch, Seq_Len, Channels) for RNN
        rnn_in = struct_features.permute(0, 2, 1)

        # BiGRU
        # rnn_out: (Batch, Seq_Len, Hidden*2)
        rnn_out, _ = self.gru(rnn_in)

        # Head
        logits = self.head(rnn_out)

        return logits
