import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResNetBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block (TCN-style).
    Uses dilated convolutions to expand receptive field efficiently.
    Cite solution_lesson_node_00011.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(DilatedResNetBlock, self).__init__()

        # Padding calculation to maintain sequence length
        padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Shortcut handling if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        return self.shortcut(residual) + out


class RNA_Net(nn.Module):
    """
    One-Hot Enhanced Hybrid Network.
    Cite solution_lesson_node_00010: One-Hot Encoding + Wider Model.
    Cite solution_lesson_node_00011: Single-Layer Dilated Blocks.

    Architecture:
    1. One-Hot Encoding for Sequence, Structure, and Loop Type.
    2. Initial Convolution to map features to hidden dimension (256).
    3. Stack of Single-Layer Dilated ResNet Blocks.
    4. Bidirectional GRU for global context aggregation.
    5. Linear Output Head.
    """

    def __init__(self):
        super(RNA_Net, self).__init__()

        # 1. Input Dimensions (One-Hot)
        # Sequence (4) + Structure (3) + Loop (7) = 14
        in_channels = (
            Config.VOCAB_SIZE_SEQ + Config.VOCAB_SIZE_STRUCT + Config.VOCAB_SIZE_LOOP
        )
        hidden_dim = Config.HIDDEN_DIM

        # 2. Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Backbone: Dilated ResNet Blocks
        layers = []
        for dilation in Config.DILATIONS:
            layers.append(
                DilatedResNetBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=Config.KERNEL_SIZE,
                    dilation=dilation,
                    dropout=Config.DROPOUT,
                )
            )
        self.backbone = nn.Sequential(*layers)

        # 4. Global Context: BiGRU
        # Hidden size is halved so that bidirectional output matches hidden_dim
        # Cite solution_lesson_node_00004
        gru_hidden = hidden_dim // 2
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )

        # 5. Output Head
        # BiGRU output dim = gru_hidden * 2 = hidden_dim
        self.head = nn.Linear(hidden_dim, Config.NUM_TARGETS)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, 3) containing integer indices.
               Channel 0: Sequence
               Channel 1: Structure
               Channel 2: Loop Type
        Returns:
            out: Tensor of shape (Batch, Seq_Len, Num_Targets)
        """
        # Separate inputs
        seq_idx = x[:, :, 0]  # (B, L)
        struct_idx = x[:, :, 1]  # (B, L)
        loop_idx = x[:, :, 2]  # (B, L)

        # One-Hot Encoding
        # (B, L) -> (B, L, Vocab_Size)
        x_seq = F.one_hot(seq_idx, num_classes=Config.VOCAB_SIZE_SEQ).float()
        x_struct = F.one_hot(struct_idx, num_classes=Config.VOCAB_SIZE_STRUCT).float()
        x_loop = F.one_hot(loop_idx, num_classes=Config.VOCAB_SIZE_LOOP).float()

        # Concatenate features
        # (B, L, 14)
        x_concat = torch.cat([x_seq, x_struct, x_loop], dim=2)

        # Permute for CNN: (B, C, L)
        x_cnn = x_concat.permute(0, 2, 1)

        # Initial Conv
        x_cnn = self.initial_conv(x_cnn)

        # Backbone
        x_cnn = self.backbone(x_cnn)

        # Permute for RNN: (B, L, C)
        x_rnn = x_cnn.permute(0, 2, 1)

        # BiGRU
        # out shape: (B, L, 2 * gru_hidden) = (B, L, hidden_dim)
        x_rnn, _ = self.gru(x_rnn)

        # Output Head
        # (B, L, Num_Targets)
        out = self.head(x_rnn)

        return out
