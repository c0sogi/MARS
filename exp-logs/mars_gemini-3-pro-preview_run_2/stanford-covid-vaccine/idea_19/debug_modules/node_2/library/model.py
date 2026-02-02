import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedDenseBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block utilizing Dense Connections.
    Applies a dilated convolution to the concatenated history of features.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedDenseBlock, self).__init__()
        # Calculate padding to maintain sequence length (same padding)
        # padding = dilation * (kernel_size - 1) // 2
        self.padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False,
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


class DenseTCN(nn.Module):
    """
    Backbone: Dense Dilated TCN.
    Stacks DilatedDenseBlocks and concatenates their outputs (Dense Connections).
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilations, dropout):
        super(DenseTCN, self).__init__()

        self.blocks = nn.ModuleList()
        self.growth_rate = growth_rate

        # Initial embedding/projection layer
        # Maps raw inputs (18 channels) to the hidden dimension (64 channels)
        self.embedding = nn.Conv1d(in_channels, growth_rate, kernel_size=1)

        # Build blocks with Dense Connections
        # The input to block i is the concatenation of the embedding and all previous block outputs.
        current_input_channels = growth_rate

        for dilation in dilations:
            block = DilatedDenseBlock(
                in_channels=current_input_channels,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)
            # Input to the next block increases by the growth rate (output of current block)
            current_input_channels += growth_rate

    def forward(self, x):
        # x: (Batch, In_Channels, SeqLen)

        # Initial embedding
        embed = self.embedding(x)  # (B, 64, L)

        # List to store all feature maps for dense connections
        features = [embed]

        # Pass through blocks
        for block in self.blocks:
            # Concatenate all previous features along channel dimension
            in_feat = torch.cat(features, dim=1)
            out = block(in_feat)
            features.append(out)

        # Output: Concatenation of all block outputs (excluding the initial embedding)
        # This creates the "High-dimensional tensor H_dense" (e.g., 6 blocks * 64 = 384 channels)
        block_outputs = features[1:]
        return torch.cat(block_outputs, dim=1)


class AsymmetricFusion(nn.Module):
    """
    Asymmetric Latent Fusion Module.
    Decouples Local Fidelity from Structural Context using a dual-stream approach
    and a Partner Gather operation.
    """

    def __init__(self, in_channels, local_dim, struct_dim):
        super(AsymmetricFusion, self).__init__()

        # Stream 1: Local Projection (High dim, preserves local details)
        self.local_proj = nn.Sequential(
            nn.Conv1d(in_channels, local_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(local_dim),
            nn.ReLU(),
        )

        # Stream 2: Structural Projection (Low dim, for message passing)
        self.struct_proj = nn.Sequential(
            nn.Conv1d(in_channels, struct_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(struct_dim),
            nn.ReLU(),
        )

    def forward(self, h_dense, partner_indices):
        """
        Args:
            h_dense: (Batch, Dense_Channels, SeqLen)
            partner_indices: (Batch, SeqLen) - Indices of paired bases, -1 if unpaired.
        """
        # Stream 1: Local features
        h_local = self.local_proj(h_dense)  # (B, local_dim, L)

        # Stream 2: Pre-gather structural features
        h_struct_pre = self.struct_proj(h_dense)  # (B, struct_dim, L)

        # Latent Gather Logic
        B, C, L = h_struct_pre.shape

        # 1. Handle -1 indices (unpaired)
        # Create a boolean mask where partner exists
        mask = partner_indices != -1  # (B, L)

        # Replace -1 with 0 to make gather safe (we will mask the result later)
        safe_indices = partner_indices.clone()
        safe_indices[~mask] = 0

        # 2. Prepare indices for gather
        # We need indices of shape (B, C, L) to gather along dim 2 (SeqLen)
        # safe_indices is (B, L) -> unsqueeze(1) -> (B, 1, L) -> expand to (B, C, L)
        gather_indices = safe_indices.unsqueeze(1).expand(-1, C, -1)

        # 3. Gather
        # h_struct_pre is (B, C, L)
        # For each batch b, channel c, position i, retrieve h_struct_pre[b, c, gather_indices[b, c, i]]
        m_partner = torch.gather(h_struct_pre, 2, gather_indices)

        # 4. Apply Mask
        # Zero out positions that were unpaired (where we gathered from index 0 erroneously)
        mask_expanded = mask.unsqueeze(1).expand(-1, C, -1)
        m_partner = m_partner * mask_expanded.float()

        # Fusion
        # Concatenate Local features and Partner Message
        fused = torch.cat([h_local, m_partner], dim=1)  # (B, local_dim + struct_dim, L)

        return fused


class RNAModel(nn.Module):
    """
    Main Model Class: Asymmetric Dense-Context Hybrid Network.
    Combines DenseTCN backbone, Asymmetric Fusion, and BiGRU aggregation.
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # Hyperparameters from Config
        self.hidden_dim = Config.HIDDEN_DIM
        self.dilations = Config.DILATIONS
        self.dropout = Config.DROPOUT
        self.kernel_size = Config.KERNEL_SIZE

        # Input Dimension Calculation
        # Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18 channels
        self.input_dim = 18

        # 1. Backbone: Dense Dilated TCN
        self.backbone = DenseTCN(
            in_channels=self.input_dim,
            growth_rate=self.hidden_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout,
        )

        # Calculate output dimension of DenseTCN
        # It returns concatenation of block outputs. Num blocks = len(dilations).
        self.dense_out_dim = len(self.dilations) * self.hidden_dim

        # 2. Asymmetric Latent Fusion
        self.fusion = AsymmetricFusion(
            in_channels=self.dense_out_dim,
            local_dim=Config.LOCAL_PROJ_DIM,
            struct_dim=Config.STRUCT_PROJ_DIM,
        )

        self.fusion_out_dim = Config.LOCAL_PROJ_DIM + Config.STRUCT_PROJ_DIM

        # 3. Global Aggregation (BiGRU)
        # Hidden size is strictly input_dim // 2 to ensure output dim matches input dim
        self.rnn_hidden = self.fusion_out_dim // 2
        self.gru = nn.GRU(
            input_size=self.fusion_out_dim,
            hidden_size=self.rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # BiGRU output size is 2 * rnn_hidden
        self.head = nn.Linear(self.rnn_hidden * 2, 5)  # Predicts all 5 targets

    def forward(self, x, partner_indices):
        """
        Args:
            x: Input features (Batch, SeqLen, Channels)
            partner_indices: Partner map (Batch, SeqLen)
        Returns:
            logits: (Batch, SeqLen, 5)
        """
        # Permute x to (Batch, Channels, SeqLen) for Conv1d operations
        x = x.permute(0, 2, 1)

        # Backbone Forward
        h_dense = self.backbone(x)  # (B, dense_out_dim, L)

        # Fusion Forward
        h_fused = self.fusion(h_dense, partner_indices)  # (B, fusion_out_dim, L)

        # Prepare for RNN: (Batch, SeqLen, Channels)
        h_fused = h_fused.permute(0, 2, 1)

        # GRU Forward
        out, _ = self.gru(h_fused)  # (B, L, rnn_hidden*2)

        # Head Forward
        logits = self.head(out)  # (B, L, 5)

        return logits
