import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block that produces 'growth_rate' new features.
    Designed to be used in a DenseNet-style architecture where input and output
    are concatenated externally.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super(DenseDilatedBlock, self).__init__()
        # Calculate padding to maintain sequence length
        # padding = (kernel_size - 1) * dilation / 2
        # Assuming odd kernel_size (e.g., 3)
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,  # Bias handled by BN
        )
        self.bn = nn.BatchNorm1d(growth_rate)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class StackingInteraction(nn.Module):
    """
    Implements the Latent Neighbor-Stacking Interaction.
    Compresses backbone features and gathers context from the paired base
    and its immediate neighbors (stacking partners).
    """

    def __init__(self, in_channels, compressed_dim, num_neighbors=3):
        super(StackingInteraction, self).__init__()
        self.num_neighbors = num_neighbors

        # Compression layer to reduce dimensionality before gathering
        self.compress = nn.Conv1d(
            in_channels, compressed_dim, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm1d(compressed_dim)
        self.act = nn.ReLU()

    def forward(self, x, neighbor_indices):
        """
        Args:
            x: Tensor of shape (Batch, Channels, SeqLen)
            neighbor_indices: Tensor of shape (Batch, SeqLen, NumNeighbors)
                              containing indices of partner, partner-1, partner+1.
        Returns:
            Tensor of shape (Batch, CompressedDim * (1 + NumNeighbors), SeqLen)
        """
        B, C, L = x.shape

        # 1. Compress features: (B, C, L) -> (B, C_comp, L)
        compressed = self.act(self.bn(self.compress(x)))
        C_comp = compressed.shape[1]

        # List to hold local + gathered features
        features_list = [compressed]

        # 2. Gather features for each neighbor type
        for k in range(self.num_neighbors):
            # Get indices for the k-th neighbor: (B, L)
            idx = neighbor_indices[:, :, k].long()

            # Create a validity mask: 1 where idx != -1, else 0
            # Shape: (B, 1, L) for broadcasting
            mask = (idx != -1).unsqueeze(1).float()

            # Replace -1 with 0 to ensure valid gathering indices
            # (We will zero out the result using the mask later)
            safe_idx = idx.clone()
            safe_idx[safe_idx == -1] = 0

            # Expand indices to match channel dimension for gather
            # Target Shape: (B, C_comp, L)
            gather_idx = safe_idx.unsqueeze(1).expand(-1, C_comp, -1)

            # Gather features along sequence dimension (dim 2)
            neighbor_feat = torch.gather(compressed, 2, gather_idx)

            # Apply mask to zero out features gathered from dummy index 0
            neighbor_feat = neighbor_feat * mask

            features_list.append(neighbor_feat)

        # 3. Concatenate: Local + Partner + Partner_Minus1 + Partner_Plus1
        # Output dim: C_comp * 4
        out = torch.cat(features_list, dim=1)
        return out


class DenseStackingHybridNet(nn.Module):
    """
    Idea 15: Dense-Stacking Hybrid Network.
    Combines a Dense Dilated TCN backbone with a physics-aware Stacking Interaction
    layer that explicitly models base-pair geometry and thermodynamics.
    """

    def __init__(self):
        super(DenseStackingHybridNet, self).__init__()

        # Hyperparameters
        input_dim = Config.INPUT_DIM
        growth_rate = Config.HIDDEN_DIM
        dilations = Config.DILATIONS
        dropout = Config.DROPOUT
        kernel_size = Config.KERNEL_SIZE
        stacking_neighbors = Config.STACKING_NEIGHBORS

        # 1. Input Projection
        # Project raw inputs to the hidden dimension (growth rate)
        self.input_proj = nn.Conv1d(input_dim, growth_rate, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm1d(growth_rate)
        self.input_act = nn.ReLU()

        # 2. Dense Dilated Backbone
        self.blocks = nn.ModuleList()
        current_dim = growth_rate

        for d in dilations:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            # In DenseNet, input to next layer is concatenation of previous input + block output
            current_dim += growth_rate

        # 3. Stacking Interaction Layer
        # We compress the high-dimensional backbone output to 'growth_rate' size
        # before gathering to keep memory usage reasonable.
        self.stacking = StackingInteraction(
            in_channels=current_dim,
            compressed_dim=growth_rate,
            num_neighbors=stacking_neighbors,
        )

        # Calculate input dimension for GRU
        # Stacking output = Compressed(Local) + 3 * Compressed(Neighbors)
        gru_input_dim = growth_rate * (1 + stacking_neighbors)

        # 4. Global Aggregation (BiGRU)
        # Hidden size is set to half of input size so that bidirectional output matches input size
        gru_hidden_dim = gru_input_dim // 2
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Head
        # BiGRU output dim = gru_hidden_dim * 2
        self.head = nn.Linear(gru_hidden_dim * 2, 5)

    def forward(self, x, neighbor_indices):
        """
        Forward pass of the model.

        Args:
            x: Input tensor (Batch, SeqLen, InputDim)
            neighbor_indices: Neighbor index map (Batch, SeqLen, 3)

        Returns:
            logits: Predicted values (Batch, SeqLen, 5)
        """
        # Permute to (Batch, Channels, SeqLen) for Conv1d
        x = x.transpose(1, 2)

        # Input Projection
        x = self.input_act(self.input_bn(self.input_proj(x)))

        # Dense Backbone
        for block in self.blocks:
            out = block(x)
            # Dense Connection: Concatenate input and output along channel dimension
            x = torch.cat([x, out], dim=1)

        # Stacking Interaction
        # Input x shape: (Batch, Total_Dense_Channels, SeqLen)
        # Output x shape: (Batch, GRU_Input_Dim, SeqLen)
        x = self.stacking(x, neighbor_indices)

        # Prepare for GRU: (Batch, SeqLen, Channels)
        x = x.transpose(1, 2)

        # BiGRU
        x, _ = self.gru(x)

        # Output Head
        logits = self.head(x)

        return logits
