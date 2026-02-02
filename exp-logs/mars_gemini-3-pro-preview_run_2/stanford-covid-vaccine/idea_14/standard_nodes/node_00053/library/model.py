import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    A single dilated residual block with dense connectivity.

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): Number of output channels (added to the dense stack).
        kernel_size (int): Convolution kernel size.
        dilation (int): Dilation rate.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(growth_rate)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(self.bn(out))
        out = self.dropout(out)
        # Dense connection: concatenate input with new features
        return torch.cat([x, out], dim=1)


class DenseDilatedTCN(nn.Module):
    """
    Dense Dilated Temporal Convolutional Network.

    Constructs a stack of DenseBlocks with exponentially increasing dilation rates.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation_rates, dropout):
        super(DenseDilatedTCN, self).__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilation_rates:
            block = DenseBlock(current_dim, growth_rate, kernel_size, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

        self.out_channels = current_dim

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class NeighborStackingGather(nn.Module):
    """
    Latent Neighbor-Stacking Interaction Module.

    Compresses high-dimensional dense features and gathers context from:
    1. The base's partner (p)
    2. The partner's 5' neighbor (p-1)
    3. The partner's 3' neighbor (p+1)

    This explicitly models the stacking geometry of RNA secondary structure.
    """

    def __init__(self, in_channels, compress_dim):
        super(NeighborStackingGather, self).__init__()
        self.compress = nn.Conv1d(in_channels, compress_dim, 1)
        self.compress_dim = compress_dim

    def forward(self, x, indices):
        """
        Args:
            x (torch.Tensor): Backbone features of shape (B, C_in, L).
            indices (dict): Dictionary containing 'p', 'pm1', 'pp1' index tensors of shape (B, L).

        Returns:
            torch.Tensor: Fused features of shape (B, L, C_in + 3 * C_compressed).
        """
        B, C_in, L = x.shape

        # 1. Compress features for gathering
        h_small = self.compress(x)  # (B, C_small, L)

        # 2. Prepare for gathering: Permute to (B, L, C_small)
        h_small_t = h_small.permute(0, 2, 1)

        # 3. Handle Padding: Append a zero vector at index L for invalid neighbors
        padding_vec = torch.zeros(
            B, 1, self.compress_dim, device=x.device, dtype=x.dtype
        )
        h_padded = torch.cat([h_small_t, padding_vec], dim=1)  # (B, L+1, C_small)

        # 4. Helper for gathering
        # We use advanced indexing.
        # Create batch indices: (B, 1) -> (B, L)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)

        def gather_tensor(idx_tensor):
            # idx_tensor is (B, L)
            # Returns (B, L, C_small)
            return h_padded[batch_idx, idx_tensor, :]

        # 5. Gather from the three spatial locations
        feat_p = gather_tensor(indices["p"])
        feat_pm1 = gather_tensor(indices["pm1"])
        feat_pp1 = gather_tensor(indices["pp1"])

        # 6. Fuse: Concatenate local full features with gathered compressed features
        # x needs permute to (B, L, C_in)
        x_t = x.permute(0, 2, 1)

        fused = torch.cat([x_t, feat_p, feat_pm1, feat_pp1], dim=2)
        return fused


class HybridNet(nn.Module):
    """
    The main Dense Latent-Neighbor Hybrid Network.

    Pipeline:
    Input -> Dense Dilated TCN -> Neighbor Stacking Gather -> BiGRU -> Output Head
    """

    def __init__(self):
        super(HybridNet, self).__init__()

        # Input Dimensions
        # Sequence (4) + Structure (3) + Loop Type (7) + Partner Seq (4) = 18
        input_dim = 18

        # Backbone Configuration
        self.backbone = DenseDilatedTCN(
            in_channels=input_dim,
            growth_rate=Config.HIDDEN_DIM,
            kernel_size=Config.KERNEL_SIZE,
            dilation_rates=Config.DILATION_RATES,
            dropout=Config.DROPOUT,
        )

        # Interaction Module
        self.gather = NeighborStackingGather(
            in_channels=self.backbone.out_channels, compress_dim=Config.HIDDEN_DIM
        )

        # Global Aggregation (BiGRU)
        # Input to GRU is: Backbone_Out + 3 * Compressed_Dim
        gru_input_dim = self.backbone.out_channels + 3 * Config.HIDDEN_DIM

        # Constraint: Hidden size is input_dim // 2
        gru_hidden_dim = gru_input_dim // 2

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        # BiGRU output is 2 * hidden_size
        head_in_dim = 2 * gru_hidden_dim
        self.head = nn.Linear(head_in_dim, 5)

    def forward(self, x, indices):
        """
        Args:
            x (torch.Tensor): Input features (B, 14, L).
            indices (dict): Structure indices for gathering.

        Returns:
            torch.Tensor: Predictions (B, L, 5).
        """
        # 1. Backbone
        # Output: (B, Backbone_Channels, L)
        features = self.backbone(x)

        # 2. Neighbor Stacking Interaction
        # Output: (B, L, Fused_Channels)
        fused = self.gather(features, indices)

        # 3. Global Aggregation
        # Output: (B, L, 2 * Hidden)
        gru_out, _ = self.gru(fused)

        # 4. Prediction Head
        logits = self.head(gru_out)

        return logits
