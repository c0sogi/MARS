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
    Multi-Scale CNN-LSTM (Hybrid) Model.

    Simplified architecture that consumes all features (raw, physics, derivatives)
    through a single deep branch. (Cite solution_lesson_node_00022)
    """

    def __init__(self):
        super().__init__()

        # Input dimension = All Continuous Features
        cnn_in_dim = len(config.CONTINUOUS_FEATURES)

        # --- Stem (Multi-Scale CNN) ---
        self.cnn_stem = MultiScaleConv1d(
            in_channels=cnn_in_dim,
            out_channels=config.CNN_FILTERS,
            kernels=config.CNN_KERNELS,
            dropout=config.CNN_DROPOUT,
        )

        # --- Backbone (LSTM) ---
        lstm_in_dim = config.CNN_FILTERS * len(config.CNN_KERNELS)

        self.lstm_backbone = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL,
            batch_first=True,
        )

        # --- Projection Head ---
        lstm_out_dim = config.LSTM_HIDDEN_SIZE * (2 if config.BIDIRECTIONAL else 1)
        self.head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x_cont, **kwargs):
        """
        Args:
            x_cont (Tensor): Continuous features (Batch, Seq, Feat_Cont)
            **kwargs: Ignored

        Returns:
            Tensor: Predicted pressure (Batch, Seq)
        """
        # Permute to (Batch, Channels, Seq) for Conv1d
        cnn_input = x_cont.permute(0, 2, 1)
        cnn_out = self.cnn_stem(cnn_input)  # (Batch, C_out, Seq)

        # Permute back to (Batch, Seq, Channels) for LSTM
        lstm_input = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm_backbone(lstm_input)  # (Batch, Seq, Hidden*Dirs)

        # Final Prediction
        output = self.head(lstm_out)  # (Batch, Seq, 1)

        return output.squeeze(-1)
