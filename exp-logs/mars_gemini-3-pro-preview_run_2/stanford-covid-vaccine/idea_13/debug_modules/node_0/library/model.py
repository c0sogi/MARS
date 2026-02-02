import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseTCNBlock(nn.Module):
    """
    A single dilated convolutional block for the Dense Backbone.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DenseTCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
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


class LatentStructuralGather(nn.Module):
    """
    Gathers features from partner positions defined by partner_indices.
    """

    def __init__(self):
        super(LatentStructuralGather, self).__init__()

    def forward(self, x, partner_indices):
        """
        Args:
            x: Tensor of shape (Batch, Channels, Seq_Len)
            partner_indices: LongTensor of shape (Batch, Seq_Len)
        Returns:
            Tensor of shape (Batch, Channels, Seq_Len) containing partner features.
        """
        batch_size, channels, seq_len = x.shape

        # Expand indices to cover all channels: (Batch, Channels, Seq_Len)
        # We need to repeat the indices along the channel dimension
        idx = partner_indices.unsqueeze(1).expand(-1, channels, -1)

        # Gather features
        partner_features = torch.gather(x, 2, idx)
        return partner_features


class StackingRefinementBlock(nn.Module):
    """
    Processes paired states to model base-pair stacking interactions.
    """

    def __init__(
        self, in_channels, hidden_channels, kernel_size=3, layers=2, dropout=0.1
    ):
        super(StackingRefinementBlock, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        padding = kernel_size // 2
        self.layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        )

        # Subsequent layers
        for _ in range(layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels, hidden_channels, kernel_size, padding=padding
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BridgedHybridNet(nn.Module):
    """
    Bridged Dense-Refined Hybrid Network (Idea 14).
    Combines a Dense Dilated TCN backbone with a Parallel Stacking Refinement branch.
    """

    def __init__(self):
        super(BridgedHybridNet, self).__init__()

        # ==========================================
        # 1. Dense Backbone (Dilated TCN)
        # ==========================================
        self.dense_blocks = nn.ModuleList()
        current_channels = Config.IN_CHANNELS
        growth_rate = Config.GROWTH_RATE

        # Build blocks with increasing dilation
        for dilation in Config.DILATIONS:
            block = DenseTCNBlock(
                in_channels=current_channels,
                out_channels=growth_rate,
                kernel_size=Config.KERNEL_SIZE,
                dilation=dilation,
                dropout=Config.DROPOUT,
            )
            self.dense_blocks.append(block)
            # In DenseNet, we concatenate output to input for the next layer
            current_channels += growth_rate

        self.backbone_out_channels = current_channels

        # ==========================================
        # 2. Parallel Stacking Refinement Branch
        # ==========================================
        # Compression 1x1 Conv
        self.compress_conv = nn.Conv1d(
            self.backbone_out_channels, growth_rate, kernel_size=1
        )

        # Structural Gather
        self.gather = LatentStructuralGather()

        # Stacking Module
        # Input to stacking is (Compressed_Self + Compressed_Partner)
        stacking_in_channels = growth_rate * 2
        self.stacking_module = StackingRefinementBlock(
            in_channels=stacking_in_channels,
            hidden_channels=growth_rate,
            kernel_size=Config.STACKING_KERNEL_SIZE,
            layers=Config.STACKING_LAYERS,
            dropout=Config.DROPOUT,
        )

        # ==========================================
        # 3. Bridged Fusion & Global Aggregation
        # ==========================================
        # Bridge: Concatenate Backbone Output + Stacking Output
        fusion_channels = self.backbone_out_channels + growth_rate

        # BiGRU
        gru_hidden_size = fusion_channels // 2
        self.gru = nn.GRU(
            input_size=fusion_channels,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ==========================================
        # 4. Output Head
        # ==========================================
        self.out_proj = nn.Linear(gru_hidden_size * 2, 5)

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs: (Batch, In_Channels, Seq_Len)
            partner_indices: (Batch, Seq_Len)
        """
        # 1. Dense Backbone Forward Pass
        # We maintain a list of all features to facilitate dense concatenation
        features = [inputs]

        for block in self.dense_blocks:
            # Concatenate all previous features
            dense_in = torch.cat(features, dim=1)
            # Compute block output
            out = block(dense_in)
            # Store output
            features.append(out)

        # Final Backbone Representation (Concatenation of all)
        h_dense = torch.cat(features, dim=1)  # (Batch, backbone_out_channels, Seq_Len)

        # 2. Stacking Refinement Branch
        # Compress
        h_compressed = self.compress_conv(h_dense)  # (Batch, growth_rate, Seq_Len)

        # Gather Partner Features
        h_partner = self.gather(h_compressed, partner_indices)

        # Form Pair State
        h_pair = torch.cat(
            [h_compressed, h_partner], dim=1
        )  # (Batch, growth_rate*2, Seq_Len)

        # Process Stacking Physics
        h_stack = self.stacking_module(h_pair)  # (Batch, growth_rate, Seq_Len)

        # 3. Bridged Fusion
        h_fused = torch.cat(
            [h_dense, h_stack], dim=1
        )  # (Batch, fusion_channels, Seq_Len)

        # 4. Global Aggregation (BiGRU)
        # Permute for RNN: (Batch, Seq_Len, Channels)
        h_fused_perm = h_fused.permute(0, 2, 1)

        gru_out, _ = self.gru(h_fused_perm)

        # 5. Output Projection
        logits = self.out_proj(gru_out)  # (Batch, Seq_Len, 5)

        # Permute back to (Batch, 5, Seq_Len) if needed, or keep as is.
        # The metric calculation expects (N, Seq, 5) or flattened.
        # Usually PyTorch Loss expects (N, C, L) or (N, L, C).
        # We return (N, Seq_Len, 5) to match common usage.

        return logits
