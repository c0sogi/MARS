import torch
import torch.nn as nn
from library.config import (
    CNN_FILTERS,
    RNN_HIDDEN_SIZE,
    RNN_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    SPEC_CHANNELS,
    FREQ_BINS,
)


class SpectrogramCRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for EEG Spectrogram Classification.

    Architecture:
    1. Encoder: 3-block 2D CNN to extract features from (Time, Freq) maps.
       - Uses aggressive pooling on the Frequency axis to reduce dimensionality.
       - Uses conservative pooling on the Time axis to preserve temporal resolution.
    2. Temporal Aggregator: Bidirectional GRU to model the evolution of features over time.
    3. Classifier: Global Average Pooling followed by a Dense layer and Softmax.
    """

    def __init__(
        self,
        input_channels: int = SPEC_CHANNELS,
        freq_bins: int = FREQ_BINS,
        cnn_filters: list = CNN_FILTERS,
        rnn_hidden_size: int = RNN_HIDDEN_SIZE,
        rnn_layers: int = RNN_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 4 regions: LL, RL, LP, RP).
            freq_bins (int): Total frequency bins across all channels (used to calc bins per channel).
            cnn_filters (list): List of filter counts for the 3 CNN blocks.
            rnn_hidden_size (int): Hidden size for the GRU layer.
            rnn_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout rate for the classifier.
            num_classes (int): Number of output classes.
        """
        super().__init__()

        # Calculate frequency bins per region (channel)
        # The input tensor is expected to be (Batch, Channels, Time, Region_Bins)
        self.region_bins = freq_bins // input_channels

        # --- Encoder (2D CNN) ---
        self.encoder_blocks = nn.ModuleList()

        # Block 1
        # Conv -> BN -> ReLU -> MaxPool (Aggressive Freq Pooling)
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels, cnn_filters[0], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(cnn_filters[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2)),  # (Time, Freq) -> (T, F/2)
            )
        )

        # Block 2
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(cnn_filters[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2)),  # (Time, Freq) -> (T, F/4)
            )
        )

        # Block 3
        # Start reducing Time dimension here as well
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(cnn_filters[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),  # (Time, Freq) -> (T/2, F/8)
            )
        )

        # --- Calculate RNN Input Size ---
        # Trace the reduction of the frequency dimension through pooling layers
        f = self.region_bins
        f = f // 2  # Block 1 pooling
        f = f // 2  # Block 2 pooling
        f = f // 2  # Block 3 pooling

        # RNN input size = Number of CNN filters * Remaining Frequency bins
        self.rnn_input_size = cnn_filters[2] * f

        # --- Temporal Aggregator (RNN) ---
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        # --- Classifier ---
        # Bidirectional RNN outputs hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size * 2, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (Batch, Channels, Time, Freq)
               Example: (32, 4, 300, 100)

        Returns:
            torch.Tensor: Class probabilities of shape (Batch, Num_Classes)
        """
        # 1. Encoder Pass
        for block in self.encoder_blocks:
            x = block(x)

        # Current shape: (Batch, Filters, Time_Reduced, Freq_Reduced)
        b, c, t, f = x.size()

        # 2. Reshape for RNN
        # Permute to (Batch, Time, Filters, Freq) to make Time the sequence dimension
        x = x.permute(0, 2, 1, 3)

        # Flatten Filters and Freq into a single feature vector per time step
        # Shape: (Batch, Time, Filters * Freq)
        x = x.reshape(b, t, c * f)

        # 3. RNN Pass
        # Output shape: (Batch, Time, Hidden * 2)
        x, _ = self.rnn(x)

        # 4. Global Average Pooling
        # Average over the Time dimension to get a single vector per sample
        # Shape: (Batch, Hidden * 2)
        x = torch.mean(x, dim=1)

        # 5. Classifier
        # Output shape: (Batch, Num_Classes)
        x = self.classifier(x)

        return x
