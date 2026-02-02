import torch
import torch.nn as nn
from library.config import Config


class TemporalBlock(nn.Module):
    """
    A single block for the Pyramidal TCN Branch.
    Consists of Dilated Conv1d -> GELU -> Dropout -> Dilated Conv1d -> GELU -> Dropout.
    Includes a residual connection with an optional 1x1 Conv projection if channels change.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.act1, self.dropout1, self.conv2, self.act2, self.dropout2
        )

        # Residual connection handling
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.GELU()

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranch(nn.Module):
    """
    Branch 1: Pyramidal Wide-Kernel TCN (The Resistive Stream).
    Models high-frequency, derivative-dependent dynamics.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Calculate padding to keep sequence length same (Centered window)
            # padding = (dilation * (kernel_size - 1)) / 2
            # We assume kernel_size is odd (e.g., 7) so this division is clean integer.
            padding = (kernel_size - 1) * dilation_size // 2

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]

    def forward(self, x):
        # x shape: [Batch, Features, Seq_Len]
        return self.network(x)


class LSTMBranch(nn.Module):
    """
    Branch 2: High-Capacity Bidirectional LSTM (The Elastic Stream).
    Models low-frequency, integral-dependent dynamics.
    """

    def __init__(
        self, input_size, hidden_size, num_layers, dropout, bidirectional=True
    ):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        output, _ = self.lstm(x)
        return output


class RPCNet(nn.Module):
    """
    Robust Pyramidal Context-Aware Network (RPC-Net).
    Combines a Pyramidal TCN and a High-Capacity Bi-LSTM to model
    resistive and elastic lung dynamics respectively.
    """

    def __init__(self):
        super(RPCNet, self).__init__()

        # --- Configuration ---
        num_features = Config.NUM_FEATURES

        # TCN Hyperparameters
        tcn_channels = Config.TCN_CHANNELS
        tcn_kernel = Config.TCN_KERNEL_SIZE
        tcn_dropout = Config.TCN_DROPOUT

        # LSTM Hyperparameters
        lstm_hidden = Config.LSTM_HIDDEN_SIZE
        lstm_layers = Config.LSTM_LAYERS
        lstm_bidir = Config.LSTM_BIDIRECTIONAL
        lstm_dropout = Config.LSTM_DROPOUT

        # Fusion Hyperparameters
        fc_hidden = Config.FC_HIDDEN_SIZE

        # --- Branches ---
        # Branch 1: TCN (Resistive)
        self.tcn_branch = TCNBranch(
            num_inputs=num_features,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
        )

        # Branch 2: LSTM (Elastic)
        self.lstm_branch = LSTMBranch(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=lstm_bidir,
        )

        # --- Fusion Head ---
        # Concatenate outputs of TCN and LSTM
        fusion_input_dim = self.tcn_branch.out_dim + self.lstm_branch.out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, fc_hidden), nn.GELU(), nn.Linear(fc_hidden, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            out: Output tensor of shape (Batch, Seq_Len, 1)
        """
        # 1. TCN Branch
        # TCN expects (Batch, Channels, Seq_Len)
        x_tcn = x.permute(0, 2, 1)
        tcn_out = self.tcn_branch(x_tcn)
        # Permute back to (Batch, Seq_Len, Channels) for concatenation
        tcn_out = tcn_out.permute(0, 2, 1)

        # 2. LSTM Branch
        # LSTM expects (Batch, Seq_Len, Features)
        lstm_out = self.lstm_branch(x)

        # 3. Fusion
        # Concatenate along the feature dimension
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # 4. Prediction
        out = self.head(combined)

        return out
