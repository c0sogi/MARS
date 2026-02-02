import torch
import torch.nn as nn
from library.config import Config


class TCNBlock(nn.Module):
    """
    A single block for the Temporal Convolutional Network.
    Implements: Conv1d -> BatchNorm -> GELU -> Dropout -> Residual Add
    Uses centered padding for non-causal processing.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # Calculate padding for 'same' length output with centered window
        # Padding = (kernel_size - 1) * dilation / 2
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if channels change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        res = x

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.downsample is not None:
            res = self.downsample(res)

        return out + res


class TAPINNet(nn.Module):
    """
    Time-Agnostic Physically-Integrated Non-Causal Network (TAPIN-Net).

    Structure:
    1. Input: (Batch, Seq_Len, Features) - Time-agnostic features (no time_step).
    2. Branch 1: Wide-Kernel TCN (Resistive Dynamics).
    3. Branch 2: High-Capacity Bi-LSTM (Elastic Dynamics / Integrator).
    4. Head: Concatenation -> MLP -> Pressure.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------------
        input_dim = Config.INPUT_DIM

        # TCN Hyperparameters
        tcn_kernels = Config.TCN_KERNEL_SIZE
        tcn_channels = Config.TCN_CHANNELS
        tcn_dropout = Config.TCN_DROPOUT

        # LSTM Hyperparameters
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        lstm_bidir = Config.LSTM_BIDIRECTIONAL

        # ----------------------------------------------------------------------
        # Branch 1: Non-Causal Wide-Kernel TCN
        # ----------------------------------------------------------------------
        # Pyramidal scaling: e.g., Input -> 64 -> 128 -> 256 -> 512
        layers = []
        in_c = input_dim
        # Dense TCN (dilation=1) as per Lesson 52.
        # Explicit derivatives are provided in features, so we prefer dense local processing.
        dilations = [1, 1, 1, 1]

        for i, out_c in enumerate(tcn_channels):
            # Use fixed dilation of 1
            d = 1

            layers.append(
                TCNBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=tcn_kernels,
                    dilation=d,
                    dropout=tcn_dropout,
                )
            )
            in_c = out_c

        self.tcn_branch = nn.Sequential(*layers)
        self.tcn_out_dim = tcn_channels[-1]

        # ----------------------------------------------------------------------
        # Branch 2: High-Capacity Bidirectional LSTM
        # ----------------------------------------------------------------------
        self.lstm_branch = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidir,
        )
        self.lstm_out_dim = lstm_hidden * 2 if lstm_bidir else lstm_hidden

        # ----------------------------------------------------------------------
        # Fusion Head
        # ----------------------------------------------------------------------
        # Concatenate outputs of both branches
        fusion_dim = self.tcn_out_dim + self.lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.GELU(), nn.Linear(512, 1)
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            out: Predicted pressure of shape (Batch, Seq_Len)
        """
        # ----------------------------------------------------------------------
        # TCN Path (Resistive)
        # ----------------------------------------------------------------------
        # Conv1d expects (Batch, Channels, Seq_Len)
        x_tcn = x.permute(0, 2, 1)
        tcn_out = self.tcn_branch(x_tcn)
        # Permute back to (Batch, Seq_Len, Channels)
        tcn_out = tcn_out.permute(0, 2, 1)

        # ----------------------------------------------------------------------
        # LSTM Path (Elastic)
        # ----------------------------------------------------------------------
        # LSTM expects (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm_branch(x)

        # ----------------------------------------------------------------------
        # Fusion
        # ----------------------------------------------------------------------
        # Concatenate along feature dimension
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # Predict
        out = self.head(combined)

        # Remove last dimension to match target shape (Batch, Seq_Len)
        return out.squeeze(-1)
