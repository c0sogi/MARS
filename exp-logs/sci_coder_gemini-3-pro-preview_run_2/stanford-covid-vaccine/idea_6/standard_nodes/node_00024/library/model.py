import torch
import torch.nn as nn
from library.config import Config


class SimpleBlock(nn.Module):
    """
    A Simple Dilated Convolutional Block with Residual Connection.
    Structure: Conv1d -> ReLU -> Dropout -> Residual
    Cite {lesson_id: solution_lesson_node_00011}
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(SimpleBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.dropout(out)
        return x + out


class OptimizedHybridNet(nn.Module):
    """
    Optimized Hybrid Network.

    Architecture:
    1. Linear Embedding
    2. Stack of Simple Dilated Residual Blocks (TCN)
    3. Multi-Scale Fusion (Concatenation of all block outputs)
    4. Bidirectional GRU
    5. Linear Output Head

    Cite {lesson_id: solution_lesson_node_00015} (Multi-Scale Fusion)
    Cite {lesson_id: solution_lesson_node_00017} (Compact Architecture)
    """

    def __init__(self, config=Config):
        super(OptimizedHybridNet, self).__init__()

        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.output_dim = config.OUTPUT_DIM
        self.num_layers = config.NUM_LAYERS
        self.kernel_size = config.KERNEL_SIZE
        self.dropout = config.DROPOUT

        # 1. Input Embedding
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)

        # 2. Dilated TCN Backbone
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2**i
            self.blocks.append(
                SimpleBlock(
                    self.hidden_dim,
                    self.kernel_size,
                    dilation,
                    self.dropout,
                )
            )

        # 3. Global Aggregation (BiGRU)
        # Input to GRU is the concatenation of all block outputs
        # Size: num_layers * hidden_dim
        self.gru_input_dim = self.num_layers * self.hidden_dim

        # We keep the GRU hidden size compact (same as backbone hidden dim)
        # Output will be 2 * hidden_dim (bidirectional)
        self.bigru = nn.GRU(
            self.gru_input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # Input is 2 * hidden_dim
        self.head = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Dim]

        # Embedding
        x = self.embedding(x)  # [Batch, Seq, Hidden]

        # Permute for Conv1d: [Batch, Hidden, Seq]
        x = x.permute(0, 2, 1)

        # Pass through TCN blocks and collect outputs
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        # Multi-Scale Fusion: Concatenate all block outputs along channel dimension
        # [Batch, Num_Layers * Hidden, Seq]
        x = torch.cat(block_outputs, dim=1)

        # Permute back for GRU: [Batch, Seq, Num_Layers * Hidden]
        x = x.permute(0, 2, 1)

        # BiGRU Aggregation
        # output: [Batch, Seq, 2 * Hidden]
        x, _ = self.bigru(x)

        # Final Prediction Head
        out = self.head(x)  # [Batch, Seq, Output_Dim]

        return out
