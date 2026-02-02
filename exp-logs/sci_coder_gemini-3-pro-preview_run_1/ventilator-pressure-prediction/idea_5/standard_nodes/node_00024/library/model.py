import torch
import torch.nn as nn
from library import config


class MultiScaleConv1d(nn.Module):
    """
    Inception-style Multi-Scale 1D Convolutional Block.
    Applies convolutions with different kernel sizes in parallel to capture
    features at different temporal resolutions (noise vs. trends).
    """

    def __init__(self, in_channels, out_channels, kernels, dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding="same"),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in kernels
            ]
        )

    def forward(self, x):
        # x shape: (Batch, In_Channels, Seq_Len)
        # Apply each branch
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along the channel dimension
        # Output shape: (Batch, Out_Channels * Num_Kernels, Seq_Len)
        return torch.cat(branch_outputs, dim=1)


class PhysicsResidualModel(nn.Module):
    """
    Hybrid Multi-Scale CNN-LSTM Model.

    Combines:
    1. Multi-Scale 1D Convolutional Stem (Cite solution_lesson_node_00015)
    2. Deep Bidirectional LSTM with Manual Residual Connections (Cite solution_lesson_node_00023)
    """

    def __init__(self):
        super().__init__()

        # Input dimension = All continuous features
        cnn_in_dim = len(config.CONTINUOUS_FEATURES)

        # 1. Multi-Scale CNN Stem
        self.cnn_stem = MultiScaleConv1d(
            in_channels=cnn_in_dim,
            out_channels=config.CNN_FILTERS,
            kernels=config.CNN_KERNELS,
            dropout=config.CNN_DROPOUT,
        )

        # 2. Deep LSTM Backbone with Residuals
        lstm_in_dim = config.CNN_FILTERS * len(config.CNN_KERNELS)
        self.lstm_layers = nn.ModuleList()

        # First layer maps from CNN output to Hidden Size
        self.lstm_layers.append(
            nn.LSTM(
                input_size=lstm_in_dim,
                hidden_size=config.LSTM_HIDDEN_SIZE,
                num_layers=1,
                bidirectional=config.BIDIRECTIONAL,
                batch_first=True,
            )
        )

        # Subsequent layers (input size = hidden size * 2 because bidirectional)
        hidden_dim = config.LSTM_HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1)

        for _ in range(config.LSTM_LAYERS - 1):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=config.LSTM_HIDDEN_SIZE,
                    num_layers=1,
                    bidirectional=config.BIDIRECTIONAL,
                    batch_first=True,
                )
            )

        self.dropout = nn.Dropout(config.LSTM_DROPOUT)

        # 3. Projection Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_cont, **kwargs):
        """
        Args:
            x_cont (Tensor): Continuous features (Batch, Seq, Feat_Cont)
        """
        # 1. CNN Stem
        # Permute to (Batch, Channels, Seq)
        x = x_cont.permute(0, 2, 1)
        x = self.cnn_stem(x)
        # Permute back to (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)

        # 2. LSTM Backbone with Residuals
        for i, lstm in enumerate(self.lstm_layers):
            output, _ = lstm(x)
            if i > 0:  # Residual connection for layers > 0
                x = x + output
            else:
                x = output
            x = self.dropout(x)

        # 3. Head
        pred = self.head(x)
        return pred.squeeze(-1)
