import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    A single dilated convolutional block that operates on the dense history of features.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout_prob=0.1):
        super(DenseDilatedBlock, self).__init__()
        # Standard Conv-ReLU-Dropout pattern
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=Config.KERNEL_SIZE,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: (Batch, In_Channels, Len)
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class AsymmetricInteractionLayer(nn.Module):
    """
    Projects dense features into two streams:
    1. High-fidelity local stream (preserved).
    2. Compressed message stream (gathered from partners).
    Applies zero-masking to messages from unpaired bases.
    """

    def __init__(self, in_channels, local_dim, message_dim):
        super(AsymmetricInteractionLayer, self).__init__()
        self.local_proj = nn.Conv1d(in_channels, local_dim, kernel_size=1)
        self.message_proj = nn.Conv1d(in_channels, message_dim, kernel_size=1)

    def forward(self, dense_features, partner_indices, inputs):
        """
        Args:
            dense_features: (B, C_dense, L)
            partner_indices: (B, L) - Indices of paired bases.
            inputs: (B, C_in, L) - Raw inputs used to identify unpaired bases (masking).
        """
        # 1. Generate Streams
        local_feat = self.local_proj(dense_features)  # (B, local_dim, L)
        message_feat = self.message_proj(dense_features)  # (B, message_dim, L)

        # 2. Gather Messages from Partners
        # Expand indices to match channel dimension of message_feat
        # partner_indices: (B, L) -> (B, 1, L) -> (B, message_dim, L)
        idx_expanded = partner_indices.unsqueeze(1).expand(-1, message_feat.size(1), -1)

        # Gather along sequence dimension (dim 2)
        gathered_message = torch.gather(message_feat, 2, idx_expanded)

        # 3. Apply Zero-Masking
        # Identify unpaired bases using the input feature "Partner Identity is None"
        # In inputs (19 channels), index 18 is the "None" one-hot feature.
        # Value is 1.0 if unpaired, 0.0 if paired.
        # inputs is (B, C_in, L)
        is_unpaired = inputs[:, 18, :]  # (B, L)

        # Create mask: 1.0 if paired (keep message), 0.0 if unpaired (zero message)
        mask = (1.0 - is_unpaired).unsqueeze(1)  # (B, 1, L)

        masked_message = gathered_message * mask

        # 4. Fuse Streams
        fused = torch.cat(
            [local_feat, masked_message], dim=1
        )  # (B, local_dim + message_dim, L)

        return fused


class RNAModel(nn.Module):
    """
    Asymmetric Bottleneck Dense-Context Network.
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # ----------------------------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------------------------
        self.in_channels = Config.NUM_NODE_FEATURES
        self.growth_rate = Config.CHANNEL_WIDTH
        self.dilations = Config.DILATIONS
        self.dropout = Config.DROPOUT

        # ----------------------------------------------------------------------
        # 1. Backbone: Dense Dilated TCN
        # ----------------------------------------------------------------------
        self.blocks = nn.ModuleList()
        current_dim = self.in_channels

        for d in self.dilations:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                out_channels=self.growth_rate,
                dilation=d,
                dropout_prob=self.dropout,
            )
            self.blocks.append(block)
            # Dense connection: output of this block is added to the history
            current_dim += self.growth_rate

        self.dense_out_dim = current_dim

        # ----------------------------------------------------------------------
        # 2. Asymmetric Interaction
        # ----------------------------------------------------------------------
        self.interaction = AsymmetricInteractionLayer(
            in_channels=self.dense_out_dim,
            local_dim=Config.LOCAL_DIM,
            message_dim=Config.MESSAGE_DIM,
        )

        self.rnn_input_dim = Config.LOCAL_DIM + Config.MESSAGE_DIM

        # ----------------------------------------------------------------------
        # 3. Global Aggregation (BiGRU)
        # ----------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=Config.GRU_HIDDEN_DIM,
            batch_first=True,
            bidirectional=True,
        )

        # ----------------------------------------------------------------------
        # 4. Output Head
        # ----------------------------------------------------------------------
        # BiGRU output is 2 * hidden_dim
        self.head = nn.Linear(Config.GRU_HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Seq_Len, Num_Features)
            partner_indices: (Batch, Seq_Len)
        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x_in = x.permute(0, 2, 1)

        # 1. Backbone (Dense Accumulation)
        features = [x_in]

        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)

            # Forward pass through block
            out = block(dense_input)

            # Store output for next layers
            features.append(out)

        # Final Dense Representation (Concatenation of input + all block outputs)
        dense_history = torch.cat(features, dim=1)  # (B, C_total, L)

        # 2. Asymmetric Interaction
        # Pass x_in to access the "Partner Identity" feature for masking
        interacted = self.interaction(
            dense_history, partner_indices, x_in
        )  # (B, 128, L)

        # 3. Global Aggregation
        # RNN expects (B, L, C)
        rnn_in = interacted.permute(0, 2, 1)
        rnn_out, _ = self.gru(rnn_in)  # (B, L, 2*Hidden)

        # 4. Head
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
