import torch
import torch.nn as nn
from library.config import Config


class InceptionBlock(nn.Module):
    """
    Inception-style block with parallel dilated convolutions of different kernel sizes.
    Designed to capture multi-scale features (transients vs smooth trends) simultaneously.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, dilation, dropout):
        super().__init__()
        self.branches = nn.ModuleList()

        # Create parallel branches for each kernel size
        for k in kernel_sizes:
            # Calculate padding for 'same' output size with dilation
            # Assuming k is odd, padding = (k - 1) * dilation / 2
            padding = (k - 1) * dilation // 2

            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
            self.branches.append(branch)

        # Fusion layer: Concatenate branches and project back to out_channels
        # Input dim is out_channels * number of branches
        concat_dim = out_channels * len(kernel_sizes)
        self.project = nn.Sequential(
            nn.Conv1d(concat_dim, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual connection
        # If dimensions change, project the identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (Batch, Channels, Length)

        # Apply parallel convolutions
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        cat = torch.cat(branch_outputs, dim=1)

        # Project and apply residual
        out = self.project(cat)
        res = self.residual(x)

        return out + res


class PyramidalTCN(nn.Module):
    """
    Pyramidal Temporal Convolutional Network branch.
    Channels double at each layer to increase capacity as receptive field grows.
    """

    def __init__(self, input_dim):
        super().__init__()
        layers = []
        current_dim = input_dim

        # Hyperparameters from Config
        channels_list = Config.TCN_CHANNELS
        kernel_sizes = Config.TCN_KERNEL_SIZES
        dropout = Config.TCN_DROPOUT

        for i, out_dim in enumerate(channels_list):
            dilation = 2**i
            layers.append(
                InceptionBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.out_dim = channels_list[-1]

    def forward(self, x):
        # x: (Batch, Length, Channels) -> Transpose for Conv1d -> (Batch, Channels, Length)
        x = x.transpose(1, 2)
        out = self.net(x)
        # Transpose back -> (Batch, Length, Channels)
        return out.transpose(1, 2)


class PITHNet(nn.Module):
    """
    Pyramidal Inception-TCN Hybrid (PITH-Net).
    Fuses a high-capacity Pyramidal Inception-TCN (Resistive Stream)
    with a deep Bidirectional LSTM (Elastic Stream).
    """

    def __init__(self):
        super().__init__()

        input_dim = Config.INPUT_DIM

        # --- Branch 1: Pyramidal Inception-TCN ---
        # Project input features to the first channel size defined in config
        self.tcn_input_proj = nn.Linear(input_dim, Config.TCN_CHANNELS[0])
        self.tcn = PyramidalTCN(Config.TCN_CHANNELS[0])

        # --- Branch 2: High-Capacity LSTM ---
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        # --- Fusion Head ---
        # Calculate fusion dimension
        tcn_out_dim = self.tcn.out_dim
        lstm_out_dim = Config.LSTM_HIDDEN_SIZE * (2 if Config.LSTM_BIDIRECTIONAL else 1)
        fusion_dim = tcn_out_dim + lstm_out_dim

        # Deep Dense MLP for fusion
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Length, Input_Dim)
        Returns:
            Pressure prediction of shape (Batch, Length)
        """
        # Branch 1: TCN
        x_tcn_in = self.tcn_input_proj(x)
        tcn_out = self.tcn(x_tcn_in)

        # Branch 2: LSTM
        lstm_out, _ = self.lstm(x)

        # Fusion: Concatenate features
        # tcn_out: (B, L, C_tcn)
        # lstm_out: (B, L, C_lstm)
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # Prediction
        pred = self.head(combined)  # (B, L, 1)

        return pred.squeeze(-1)
