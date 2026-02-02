import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Single-Layer Dilated Residual Block.
    Performs: Output = Input + Dropout(ReLU(Conv1d(Input)))
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualBlock, self).__init__()

        # Calculate padding to ensure 'same' output length for odd kernel sizes
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return residual + out


class ScalePartitionedDenseNet(nn.Module):
    """
    Idea 20: Scale-Partitioned Dense Hybrid Network.

    Features:
    1. Dense Dilated TCN Backbone: Concatenates outputs of all layers.
    2. Scale Partitioning: Splits dense history into Local and Global scales.
    3. Partitioned Latent Gather: Compresses and gathers partner features separately for each scale.
    4. BiGRU Aggregation: Fuses local history + partner context for final prediction.
    """

    def __init__(self):
        super(ScalePartitionedDenseNet, self).__init__()

        # ==========================================
        # Configuration
        # ==========================================
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_rate = Config.DROPOUT
        self.kernel_size = Config.KERNEL_SIZE
        self.dilation_rates = Config.DILATION_RATES
        self.partition_idx = Config.PARTITION_SPLIT_INDEX
        self.compress_dim = Config.COMPRESSION_CHANNELS

        # Input Dim: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
        self.input_dim = 18

        # ==========================================
        # Architecture
        # ==========================================

        # 1. Input Projection
        # Projects raw features to the hidden dimension
        self.input_proj = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)

        # 2. Backbone: Stack of Dilated Residual Blocks
        self.blocks = nn.ModuleList()
        for rate in self.dilation_rates:
            block = DilatedResidualBlock(
                channels=self.hidden_dim,
                kernel_size=self.kernel_size,
                dilation=rate,
                dropout=self.dropout_rate,
            )
            self.blocks.append(block)

        # 3. Scale-Partitioned Compression
        # We calculate the number of channels in the Local and Global partitions
        # based on the dense connection (concatenation of block outputs).
        num_local_blocks = self.partition_idx
        num_global_blocks = len(self.dilation_rates) - self.partition_idx

        self.local_channels = num_local_blocks * self.hidden_dim
        self.global_channels = num_global_blocks * self.hidden_dim

        # Independent compressors for each scale
        self.local_compressor = nn.Conv1d(
            self.local_channels, self.compress_dim, kernel_size=1
        )
        self.global_compressor = nn.Conv1d(
            self.global_channels, self.compress_dim, kernel_size=1
        )

        # 4. Fusion Dimensions
        # We concatenate:
        #   - The full dense history of the current base (H_dense)
        #   - The gathered compressed local features of the partner (Z_local)
        #   - The gathered compressed global features of the partner (Z_global)
        self.total_dense_channels = (
            num_local_blocks + num_global_blocks
        ) * self.hidden_dim
        self.fused_dim = (
            self.total_dense_channels + self.compress_dim + self.compress_dim
        )

        # 5. Global Aggregation (BiGRU)
        # Hidden size is set to half of input to maintain dimension size in output
        self.gru_hidden_dim = self.fused_dim // 2
        self.gru = nn.GRU(
            input_size=self.fused_dim,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # 6. Output Head
        # Maps from BiGRU output size (fused_dim) to 5 target columns
        self.head = nn.Linear(self.fused_dim, 5)

    def forward(self, x, partner_indices):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input features. Shape (Batch, SeqLen, Channels=18).
            partner_indices (torch.Tensor): Indices of paired bases. Shape (Batch, SeqLen).

        Returns:
            torch.Tensor: Predictions. Shape (Batch, SeqLen, 5).
        """
        # Permute to (Batch, Channels, SeqLen) for Conv1d operations
        x = x.permute(0, 2, 1)

        # 1. Projection
        x = self.input_proj(x)

        # 2. Backbone Pass (Dense Connections)
        block_outputs = []
        current_features = x

        for block in self.blocks:
            current_features = block(current_features)
            block_outputs.append(current_features)

        # Concatenate all block outputs to form the Dense Representation
        # Shape: (Batch, Total_Dense_Channels, SeqLen)
        h_dense = torch.cat(block_outputs, dim=1)

        # 3. Partitioning
        # Split the list of outputs into Local and Global groups
        local_outputs = block_outputs[: self.partition_idx]
        global_outputs = block_outputs[self.partition_idx :]

        # Concatenate within partitions
        h_local = torch.cat(local_outputs, dim=1)
        h_global = torch.cat(global_outputs, dim=1)

        # 4. Compression
        # Compress high-dimensional partitions to compact latent vectors
        z_local = self.local_compressor(h_local)  # (Batch, 32, SeqLen)
        z_global = self.global_compressor(h_global)  # (Batch, 32, SeqLen)

        # 5. Latent Gather
        # Gather the compressed features from the partner positions.
        # partner_indices[b, i] gives the index j of the partner of base i.
        # We want z_gathered[b, :, i] = z[b, :, j].

        def gather_features(z, indices):
            B, C, L = z.shape
            # Expand indices to match channel dimension: (Batch, Channels, SeqLen)
            indices_expanded = indices.unsqueeze(1).expand(-1, C, -1)
            # Gather along the sequence dimension (dim 2)
            return torch.gather(z, 2, indices_expanded)

        z_local_gathered = gather_features(z_local, partner_indices)
        z_global_gathered = gather_features(z_global, partner_indices)

        # 6. Fusion
        # Concatenate current base's history with partner's gathered features
        fused = torch.cat([h_dense, z_local_gathered, z_global_gathered], dim=1)

        # 7. Global Aggregation
        # Permute back to (Batch, SeqLen, Channels) for RNN
        fused = fused.permute(0, 2, 1)

        gru_out, _ = self.gru(fused)

        # 8. Output Head
        logits = self.head(gru_out)

        return logits
