import torch
import torch.nn as nn
from library.config import Config


class MaskedBiGRU(nn.Module):
    """
    Masked-Reconstruction Regularized BiGRU Model.

    This model implements a multi-task learning architecture:
    1.  Main Task: Regression of RNA degradation rates.
    2.  Auxiliary Task: Reconstruction of masked input features (Sequence, Structure, Loop Type).

    Architecture:
    - Input: (Batch, Seq_Len, 14)
    - Stem: Conv1d (Kernel=3) -> GELU
    - Backbone: Bidirectional GRU (2 layers)
    - Head 1: Linear -> 5 targets (Regression)
    - Head 2: Linear -> 14 features (Reconstruction)
    """

    def __init__(self):
        super(MaskedBiGRU, self).__init__()

        # Retrieve hyperparameters from Config
        self.input_dim = Config.INPUT_DIM  # 14
        self.hidden_dim = Config.HIDDEN_DIM  # 256
        self.num_layers = Config.NUM_LAYERS  # 2
        self.dropout = Config.DROPOUT  # 0.3
        self.bidirectional = Config.BIDIRECTIONAL  # True
        self.cnn_filters = Config.CNN_FILTERS  # 256
        self.cnn_kernel = Config.CNN_KERNEL_SIZE  # 3
        self.output_dim = Config.OUTPUT_DIM  # 5

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embeddings and aggregates local context.
        # Padding ensures output length matches input length.
        padding = (self.cnn_kernel - 1) // 2
        self.stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.cnn_filters,
            kernel_size=self.cnn_kernel,
            padding=padding,
        )
        self.activation = nn.GELU()

        # 2. BiGRU Backbone
        # Captures global sequence dependencies.
        self.gru = nn.GRU(
            input_size=self.cnn_filters,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # Determine the dimension of the GRU output
        self.gru_output_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )

        # 3. Regression Head (Main Task)
        # Predicts: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        self.regression_head = nn.Linear(self.gru_output_dim, self.output_dim)

        # 4. Reconstruction Head (Auxiliary Task)
        # Predicts the original one-hot encoded vectors for masked positions.
        self.reconstruction_head = nn.Linear(self.gru_output_dim, self.input_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, 14).

        Returns:
            reg_out (torch.Tensor): Regression predictions of shape (Batch, Seq_Len, 5).
            recon_out (torch.Tensor): Reconstruction logits of shape (Batch, Seq_Len, 14).
        """
        # Input x: (Batch, Seq_Len, Channels=14)

        # Permute for Conv1d: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)

        # Apply Stem
        x = self.stem(x)
        x = self.activation(x)

        # Permute back for GRU: (Batch, Seq_Len, Channels=256)
        x = x.permute(0, 2, 1)

        # Apply Backbone
        # gru_out: (Batch, Seq_Len, Hidden_Dim * Directions)
        gru_out, _ = self.gru(x)

        # Apply Heads
        reg_out = self.regression_head(gru_out)
        recon_out = self.reconstruction_head(gru_out)

        return reg_out, recon_out
