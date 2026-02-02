import torch
import torch.nn as nn
from library.config import Config


class MultiScaleStem(nn.Module):
    """
    Multi-Scale Convolutional Stem (Inception-1D).
    Extracts local features using parallel convolutions with different kernel sizes.
    """

    def __init__(self, input_dim, filters, kernels):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            # Calculate padding to keep sequence length same: (k - 1) / 2
            # Assumes odd kernel sizes
            pad = (k - 1) // 2
            self.convs.append(nn.Conv1d(input_dim, filters, kernel_size=k, padding=pad))

        self.act = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        # Permute for Conv1d: (Batch, Feat, Seq)
        x = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            # Apply conv and activation
            outs.append(self.act(conv(x)))

        # Concatenate along channel dimension
        # Output: (Batch, Filters * len(kernels), Seq)
        out = torch.cat(outs, dim=1)

        # Permute back: (Batch, Seq, Channels)
        out = out.transpose(1, 2)
        return out


class ResidualBiLSTMBlock(nn.Module):
    """
    Bidirectional LSTM Block with Residual Connection.
    Structure: x_{l+1} = x_l + Dropout(LSTM(x_l))
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional outputs 2 * hidden_dim
        output_dim = hidden_dim * 2

        self.dropout = nn.Dropout(dropout)

        # Projection layer for residual connection if dimensions change
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # LSTM
        h, _ = self.lstm(x)  # h: (Batch, Seq, Hidden*2)

        # Dropout
        h = self.dropout(h)

        # Residual Connection
        # Project input x to match h shape
        res = self.projection(x)

        return res + h


class MultiScaleResidualLSTM(nn.Module):
    """
    Main Model Architecture:
    1. Multi-Scale CNN Stem
    2. Stack of Residual Bi-LSTMs
    3. Regression Head
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Stem
        self.stem = MultiScaleStem(
            input_dim=config.INPUT_DIM,
            filters=config.CNN_FILTERS,
            kernels=config.CNN_KERNELS,
        )

        # Calculate output dimension of stem
        stem_out_dim = config.CNN_FILTERS * len(config.CNN_KERNELS)

        # 2. Backbone
        self.layers = nn.ModuleList()
        current_dim = stem_out_dim

        for _ in range(config.LSTM_LAYERS):
            self.layers.append(
                ResidualBiLSTMBlock(
                    input_dim=current_dim,
                    hidden_dim=config.LSTM_HIDDEN,
                    dropout=config.DROPOUT,
                )
            )
            # Subsequent layers take the output of the previous BiLSTM
            current_dim = config.LSTM_HIDDEN * 2

        # 3. Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Features)

        # Stem
        x = self.stem(x)

        # Backbone
        for layer in self.layers:
            x = layer(x)

        # Head
        # x: (Batch, Seq, Hidden*2) -> (Batch, Seq, 1)
        out = self.head(x)

        return out
