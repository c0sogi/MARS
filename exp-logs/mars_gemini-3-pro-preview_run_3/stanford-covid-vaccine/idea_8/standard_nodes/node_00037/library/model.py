import torch
import torch.nn as nn
from library.config import Config


class SpatiallyAugmentedBiGRU(nn.Module):
    """
    Spatially-Augmented Convolutional BiGRU.

    This model implements the architecture defined in the 'Spatially-Augmented Convolutional BiGRU'
    strategy. It combines a convolutional stem for local feature aggregation with a bidirectional
    GRU backbone for long-range dependency modeling.

    Architecture:
    1. Input: Spatially augmented features (Batch, Seq, 28).
       - 14 Base features (Seq + Struct + Loop)
       - 14 Paired features (Features of the base paired in 3D structure)
    2. Stem: 1D Convolution (Kernel 3, Filters 256) + GELU + Dropout.
       - Projects sparse binary features to dense context-aware embeddings.
       - 'Linearizes' local context before recurrent processing.
    3. Backbone: 2-layer Bidirectional GRU (Hidden 256).
       - Captures sequential dependencies.
    4. Head: Linear projection to 5 targets.
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing model hyperparameters.
        """
        super(SpatiallyAugmentedBiGRU, self).__init__()

        self.input_dim = config.INPUT_DIM
        self.conv_filters = config.CONV_FILTERS
        self.kernel_size = config.KERNEL_SIZE
        self.hidden_dim = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_prob = config.DROPOUT
        self.num_classes = config.NUM_CLASSES

        # --- Convolutional Stem ---
        # We use padding to maintain the sequence length (107) throughout the network.
        # Padding = (Kernel_Size - 1) / 2
        padding = (self.kernel_size - 1) // 2

        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.conv_filters,
                kernel_size=self.kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            # Removed Dropout to match the "cleaner" stem from Lesson 00031
        )

        # --- Recurrent Backbone ---
        # Bidirectional GRU
        # Dropout is applied between layers (if num_layers > 1)
        self.gru = nn.GRU(
            input_size=self.conv_filters,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )

        # --- Output Head ---
        # Projects the hidden states to the target values.
        # Input dimension is Hidden_Dim * 2 because the GRU is bidirectional.
        self.head = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Num_Classes).
        """
        # 1. Permute for Conv1d
        # Conv1d expects (Batch, Channels, Length), but input is (Batch, Length, Channels)
        x = x.permute(0, 2, 1)

        # 2. Apply Convolutional Stem
        x = self.conv_stem(x)

        # 3. Permute back for GRU
        # GRU expects (Batch, Length, Channels)
        x = x.permute(0, 2, 1)

        # 4. Apply GRU Backbone
        # output shape: (Batch, Seq_Len, Hidden_Dim * 2)
        # We ignore the hidden state (h_n) as we need predictions for every position
        x, _ = self.gru(x)

        # 5. Apply Output Head
        # output shape: (Batch, Seq_Len, Num_Classes)
        x = self.head(x)

        return x
