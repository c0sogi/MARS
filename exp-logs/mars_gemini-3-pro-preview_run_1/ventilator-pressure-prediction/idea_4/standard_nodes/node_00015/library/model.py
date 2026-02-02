import torch
import torch.nn as nn
from library.config import Config


class MultiScaleConv1d(nn.Module):
    """
    Applies parallel 1D convolutions with different kernel sizes to capture
    features at multiple temporal scales (Inception-style).
    """

    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            # Padding = (k - 1) // 2 ensures output length equals input length
            self.convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2
                )
            )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        # Apply each conv and concatenate along the channel dimension
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)


class PhysicsResidualNet(nn.Module):
    """
    Physics-Residual Multi-Scale CNN-LSTM Architecture.

    Structure:
    1. Multi-Scale 1D CNN Stem
    2. Stacked Bidirectional LSTMs with Residual Connections
    3. Linear Projection Head
    4. Physics-Informed Residual Connection (Output = Net(x) + Theoretical_Pressure)
    """

    def __init__(self):
        super().__init__()

        # --- Configuration ---
        input_dim = Config.input_dim
        cnn_filters = Config.cnn_filters
        kernel_sizes = Config.cnn_kernel_sizes
        cnn_dropout = Config.cnn_dropout

        lstm_input_size = Config.lstm_input_size
        lstm_hidden_size = Config.lstm_hidden_size
        lstm_layers = Config.lstm_layers
        bidirectional = Config.bidirectional
        lstm_dropout = Config.lstm_dropout

        # --- 1. Multi-Scale Stem ---
        self.stem = MultiScaleConv1d(input_dim, cnn_filters, kernel_sizes)
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(cnn_dropout)

        # --- 2. Residual Bi-LSTM Backbone ---
        self.lstm_layers = nn.ModuleList()
        self.lstm_dropouts = nn.ModuleList()

        # Layer 0: Adapter Layer (Input Size -> Hidden Size)
        # Note: If bidirectional, output dim is hidden_size * 2
        self.lstm_layers.append(
            nn.LSTM(
                lstm_input_size,
                lstm_hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
            )
        )
        self.lstm_dropouts.append(nn.Dropout(lstm_dropout))

        # Layers 1..N: Residual Layers (Hidden Size -> Hidden Size)
        lstm_output_dim = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        for _ in range(1, lstm_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    lstm_output_dim,
                    lstm_hidden_size,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            self.lstm_dropouts.append(nn.Dropout(lstm_dropout))

        # --- 3. Prediction Head ---
        self.head = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, Seq_Len, Features)

        Returns:
            Predicted pressure (Batch, Seq_Len)
        """
        # --- Stem Processing ---
        # Permute to (Batch, Channels, Length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)

        # Permute back to (Batch, Length, Channels) for LSTM
        x = x.permute(0, 2, 1)

        # --- LSTM Stack with Residuals ---
        for i, (lstm, drop) in enumerate(zip(self.lstm_layers, self.lstm_dropouts)):
            out, _ = lstm(x)
            out = drop(out)

            # Apply residual connection for all layers after the first
            # (First layer changes dimension, so direct add isn't possible without projection)
            if i > 0:
                x = x + out
            else:
                x = out

        # --- Head ---
        # x shape: (Batch, Seq_Len, Hidden*2)
        out = self.head(x)  # (Batch, Seq_Len, 1)
        out = out.squeeze(-1)  # (Batch, Seq_Len)

        return out
