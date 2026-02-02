import torch
import torch.nn as nn
from library.config import Config


class PyramidalTCNBlock(nn.Module):
    """
    A single block for the Pyramidal TCN branch.
    Includes dilated convolution, batch norm, activation, dropout,
    and a residual connection (with projection if channel dimensions change).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # Calculate padding to keep sequence length constant
        # Padding = (dilation * (kernel_size - 1)) / 2
        # Config.KERNEL_SIZE is typically odd (e.g., 5), ensuring integer padding.
        padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,  # Bias handled by BatchNorm
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection handling
        # If input and output channels differ, we project the identity path
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class FCPNet(nn.Module):
    """
    Fully Contextualized Pyramidal Hybrid (FCP-Net).

    Architecture:
    1. Input: Features (including PID & Physics)
    2. Branch 1: Pyramidal TCN (Resistive dynamics)
       - Increasing channels and dilation factors to capture multi-scale physics.
    3. Branch 2: Deep BiLSTM (Elastic dynamics)
       - Captures long-term dependencies and integral states.
    4. Head: Fusion of latent states (No raw input skip connection).
    """

    def __init__(self, config=Config):
        super().__init__()

        input_dim = config.INPUT_DIM
        tcn_dim = config.TCN_DIM
        lstm_dim = config.LSTM_DIM
        kernel_size = config.KERNEL_SIZE
        dropout = config.DROPOUT
        lstm_layers = config.LSTM_LAYERS

        # -------------------------------------------------------
        # Branch 1: Pyramidal Wide-Kernel TCN
        # Scaling Strategy:
        #   Layer 1: Input -> 64,  Dilation 1
        #   Layer 2: 64 -> 128,    Dilation 2
        #   Layer 3: 128 -> 256,   Dilation 4
        #   Layer 4: 256 -> 512,   Dilation 8
        # -------------------------------------------------------
        self.tcn_layers = nn.ModuleList(
            [
                PyramidalTCNBlock(
                    input_dim, tcn_dim, kernel_size, dilation=1, dropout=dropout
                ),
                PyramidalTCNBlock(
                    tcn_dim, tcn_dim * 2, kernel_size, dilation=2, dropout=dropout
                ),
                PyramidalTCNBlock(
                    tcn_dim * 2,
                    tcn_dim * 4,
                    kernel_size,
                    dilation=4,
                    dropout=dropout,
                ),
                PyramidalTCNBlock(
                    tcn_dim * 4,
                    tcn_dim * 8,
                    kernel_size,
                    dilation=8,
                    dropout=dropout,
                ),
            ]
        )

        # Final TCN output dimension (Base * 8)
        tcn_out_dim = tcn_dim * 8

        # -------------------------------------------------------
        # Branch 2: Deep Bidirectional LSTM
        # -------------------------------------------------------
        # Cite solution_lesson_node_00032: Prioritize RNN capacity for integral states
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # BiLSTM output dimension (Hidden * 2 directions)
        lstm_out_dim = lstm_dim * 2

        # -------------------------------------------------------
        # Fusion Head
        # -------------------------------------------------------
        # Concatenate TCN and LSTM latent states
        fusion_dim = tcn_out_dim + lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # --- TCN Branch ---
        # Permute to (batch, channels, seq_len) for Conv1d
        x_tcn = x.permute(0, 2, 1)

        for layer in self.tcn_layers:
            x_tcn = layer(x_tcn)

        # Permute back to (batch, seq_len, channels)
        x_tcn = x_tcn.permute(0, 2, 1)

        # --- LSTM Branch ---
        # LSTM expects (batch, seq_len, input_dim)
        x_lstm, _ = self.lstm(x)

        # --- Fusion ---
        # Concatenate along the feature dimension
        # Shape: (batch, seq_len, tcn_dim + lstm_dim)
        combined = torch.cat([x_tcn, x_lstm], dim=2)

        # Predict
        # Shape: (batch, seq_len, 1)
        out = self.head(combined)

        return out
