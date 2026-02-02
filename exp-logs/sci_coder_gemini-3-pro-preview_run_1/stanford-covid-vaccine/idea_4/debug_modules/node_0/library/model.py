import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResNetBlock1D(nn.Module):
    """
    A 1D Residual Block consisting of two convolutional layers with
    Batch Normalization and ReLU activation.
    """

    def __init__(self, channels, kernel_size, dropout=0.0):
        super(ResNetBlock1D, self).__init__()
        # Padding to maintain sequence length: (k-1)//2 for odd k
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.act2(out)

        return out


class HybridCNNBiGRU(nn.Module):
    """
    Hybrid architecture combining:
    1. Embeddings for Sequence, Structure, and Loop Type.
    2. 1D-ResNet for local motif extraction.
    3. Bidirectional GRU for global context aggregation.
    4. MLP Head for regression targets.
    """

    def __init__(self):
        super(HybridCNNBiGRU, self).__init__()

        # ----------------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------------
        self.seq_embedding = nn.Embedding(Config.SIZE_VOCAB_SEQ, Config.EMBED_DIM)
        self.struct_embedding = nn.Embedding(Config.SIZE_VOCAB_STRUCT, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.SIZE_VOCAB_LOOP, Config.EMBED_DIM)

        # Total channels after concatenation
        # 3 inputs * EMBED_DIM
        input_channels = 3 * Config.EMBED_DIM

        # ----------------------------------------------------------------
        # 2. CNN Stage (1D-ResNet)
        # ----------------------------------------------------------------
        # Projection layer to match CNN_FILTERS dimension
        self.conv_project = nn.Conv1d(
            in_channels=input_channels, out_channels=Config.CNN_FILTERS, kernel_size=1
        )

        # Stack of Residual Blocks
        self.resnet_blocks = nn.ModuleList(
            [
                ResNetBlock1D(
                    channels=Config.CNN_FILTERS,
                    kernel_size=Config.CNN_KERNEL_SIZE,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.CNN_BLOCKS)
            ]
        )

        # ----------------------------------------------------------------
        # 3. RNN Stage (Bi-GRU)
        # ----------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=Config.CNN_FILTERS,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.RNN_LAYERS > 1 else 0.0,
        )

        # ----------------------------------------------------------------
        # 4. Output Head
        # ----------------------------------------------------------------
        # Input to linear layer is hidden_dim * 2 (bidirectional)
        rnn_out_dim = Config.RNN_HIDDEN_DIM * 2

        self.fc_out = nn.Linear(rnn_out_dim, Config.NUM_TARGETS)

    def forward(self, sequence, structure, predicted_loop_type):
        """
        Args:
            sequence: (Batch, Seq_Len) LongTensor
            structure: (Batch, Seq_Len) LongTensor
            predicted_loop_type: (Batch, Seq_Len) LongTensor

        Returns:
            (Batch, Seq_Len, Num_Targets) FloatTensor
        """
        # 1. Embed inputs
        # (Batch, Seq_Len, Embed_Dim)
        emb_seq = self.seq_embedding(sequence)
        emb_struct = self.struct_embedding(structure)
        emb_loop = self.loop_embedding(predicted_loop_type)

        # Concatenate along feature dimension
        # (Batch, Seq_Len, 3 * Embed_Dim)
        x = torch.cat([emb_seq, emb_struct, emb_loop], dim=2)

        # 2. CNN Stage
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # Project to filter dimension
        x = self.conv_project(x)

        # Pass through ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)

        # 3. RNN Stage
        # Permute back to (Batch, Seq_Len, Channels) for GRU
        x = x.permute(0, 2, 1)

        # GRU output: (Batch, Seq_Len, Num_Directions * Hidden_Size)
        # We ignore the final hidden state (h_n)
        x, _ = self.gru(x)

        # 4. Output Head
        # Project to targets: (Batch, Seq_Len, Num_Targets)
        out = self.fc_out(x)

        return out
