import torch
import torch.nn as nn
from library.config import Config


class SpatiallyAugmentedBiGRU(nn.Module):
    """
    A neural network architecture for RNA degradation prediction.
    It utilizes a 1D Convolutional stem for local feature extraction and
    a Bidirectional GRU backbone for sequence modeling.
    """

    def __init__(self):
        super(SpatiallyAugmentedBiGRU, self).__init__()

        # ==========================================
        # 1. Convolutional Stem
        # ==========================================
        # Projects sparse binary features (28 dim) into a dense embedding space (256 dim).
        # We use padding to preserve the sequence length of 107.
        self.conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            padding=(Config.CONV_KERNEL_SIZE - 1) // 2,
        )

        self.act = nn.GELU()

        # ==========================================
        # 2. Recurrent Backbone (BiGRU)
        # ==========================================
        # Captures long-range dependencies along the RNA sequence.
        self.gru = nn.GRU(
            input_size=Config.CONV_FILTERS,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0.0,
            bidirectional=Config.BIDIRECTIONAL,
            batch_first=True,
        )

        # ==========================================
        # 3. Output Head
        # ==========================================
        # Projects the recurrent hidden states to the 5 target values.
        # The input dimension is doubled because the GRU is bidirectional.
        rnn_output_dim = Config.RNN_HIDDEN_DIM * (2 if Config.BIDIRECTIONAL else 1)

        self.fc = nn.Linear(rnn_output_dim, Config.OUTPUT_DIM)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).
                              Example: (64, 107, 28)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Output_Dim).
                          Example: (64, 107, 5)
        """
        # 1. Convolutional Stem
        # Conv1d expects (Batch, Channels, Length), so we transpose inputs.
        x = x.transpose(1, 2)  # (B, 107, 28) -> (B, 28, 107)
        x = self.conv(x)
        x = self.act(x)

        # Transpose back for RNN: (B, 256, 107) -> (B, 107, 256)
        x = x.transpose(1, 2)

        # 2. Recurrent Backbone
        # GRU returns (output, h_n). We only need the full sequence output.
        self.gru.flatten_parameters()  # Optimization for RNNs on GPU
        x, _ = self.gru(x)

        # 3. Output Head
        # Apply Linear layer to each time step
        x = self.fc(x)

        return x
