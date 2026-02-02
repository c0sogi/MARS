import torch
import torch.nn as nn
from library.config import Config


class BiGRURegressor(nn.Module):
    """
    Standard BiGRU Regressor with Convolutional Stem.
    Cite solution_lesson_node_00027: Convolutional Input Stems for Sparse Sequence Data.
    Cite solution_lesson_node_00044: Simultaneous Masked Reconstruction Offers No Gain.

    Architecture:
    - Input: (Batch, Seq_Len, 14)
    - Stem: Conv1d (Kernel=3) -> GELU
    - Backbone: Bidirectional GRU (2 layers)
    - Head: Linear -> 5 targets (Regression)
    """

    def __init__(self):
        super(BiGRURegressor, self).__init__()

        # Retrieve hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT
        self.bidirectional = Config.BIDIRECTIONAL
        self.cnn_filters = Config.CNN_FILTERS
        self.cnn_kernel = Config.CNN_KERNEL_SIZE
        self.output_dim = Config.OUTPUT_DIM

        # 1. Convolutional Stem
        padding = (self.cnn_kernel - 1) // 2
        self.stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.cnn_filters,
            kernel_size=self.cnn_kernel,
            padding=padding,
        )
        self.activation = nn.GELU()

        # 2. BiGRU Backbone
        self.gru = nn.GRU(
            input_size=self.cnn_filters,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.gru_output_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )

        # 3. Regression Head
        self.regression_head = nn.Linear(self.gru_output_dim, self.output_dim)

    def forward(self, x):
        # Input x: (Batch, Seq_Len, Channels=14)
        x = x.permute(0, 2, 1)

        x = self.stem(x)
        x = self.activation(x)

        x = x.permute(0, 2, 1)

        gru_out, _ = self.gru(x)

        reg_out = self.regression_head(gru_out)

        return reg_out
