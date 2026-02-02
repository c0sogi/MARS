import torch
import torch.nn as nn
from library.config import Config


class TCNBlock(nn.Module):
    """
    A single block for the Pyramidal TCN branch.
    Consists of Conv1d -> BatchNorm -> GELU -> Dropout.
    Uses centered padding to maintain sequence length (Non-Causal).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super(TCNBlock, self).__init__()
        # Calculate padding for centered convolution: p = (k - 1) / 2
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,  # Bias handled by BatchNorm
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class LANNet(nn.Module):
    """
    Lookahead-Augmented Non-Causal Hybrid (LAN-Net).

    Structure:
    1. Input: Concatenated features (PID, Physics, Lookahead).
    2. Branch 1: Pyramidal TCN (Resistive Dynamics).
    3. Branch 2: High-Capacity Bi-LSTM (Elastic Dynamics).
    4. Head: Concatenation -> Dense MLP -> Pressure Prediction.
    """

    def __init__(self, config=Config):
        super(LANNet, self).__init__()

        # ----------------------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------------------
        input_dim = config.INPUT_DIM

        # TCN Hyperparameters
        tcn_channels = config.TCN_CHANNELS
        kernel_size = config.TCN_KERNEL_SIZE
        tcn_dropout = config.TCN_DROPOUT

        # LSTM Hyperparameters
        lstm_hidden = config.LSTM_HIDDEN_SIZE
        lstm_layers = config.LSTM_LAYERS
        lstm_bidir = config.LSTM_BIDIRECTIONAL

        # ----------------------------------------------------------------------
        # Branch 1: Non-Causal Pyramidal TCN
        # ----------------------------------------------------------------------
        tcn_layers = []
        current_dim = input_dim

        for out_dim in tcn_channels:
            tcn_layers.append(TCNBlock(current_dim, out_dim, kernel_size, tcn_dropout))
            current_dim = out_dim

        self.tcn_branch = nn.Sequential(*tcn_layers)
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

        lstm_out_dim = lstm_hidden * (2 if lstm_bidir else 1)

        # ----------------------------------------------------------------------
        # Fusion Head
        # ----------------------------------------------------------------------
        # Concatenate outputs from both branches
        fusion_dim = self.tcn_out_dim + lstm_out_dim

        # Deep Dense MLP for final prediction
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for Linear and Conv layers.
        """
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
        # TCN Forward Pass
        # ----------------------------------------------------------------------
        # Conv1d expects (Batch, Channels, Seq_Len)
        x_tcn = x.permute(0, 2, 1)
        x_tcn = self.tcn_branch(x_tcn)
        # Permute back to (Batch, Seq_Len, Channels)
        x_tcn = x_tcn.permute(0, 2, 1)

        # ----------------------------------------------------------------------
        # LSTM Forward Pass
        # ----------------------------------------------------------------------
        # LSTM expects (Batch, Seq_Len, Features)
        self.lstm_branch.flatten_parameters()  # Optimization for GPU
        x_lstm, _ = self.lstm_branch(x)

        # ----------------------------------------------------------------------
        # Fusion & Prediction
        # ----------------------------------------------------------------------
        # Concatenate along feature dimension
        x_fused = torch.cat([x_tcn, x_lstm], dim=2)

        # Project to scalar output
        out = self.head(x_fused)

        # Remove last dimension to return (Batch, Seq_Len)
        return out.squeeze(-1)
