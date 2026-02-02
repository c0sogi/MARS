import torch
import torch.nn as nn
from library.config import Config


class PyramidalTCNBlock(nn.Module):
    """
    A single block of the Pyramidal TCN.

    Features:
    - Dilated 1D Convolution with Large Kernel (7)
    - Centered Padding (Non-Causal)
    - GELU Activation
    - Residual Connection (with 1x1 projection if channels change)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # Calculate padding for "Centered Padding" to maintain sequence length
        # padding = dilation * (kernel_size - 1) / 2
        # For kernel_size=7, (k-1)=6, so padding is 3 * dilation.
        padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # Projection for residual connection if dimensions change
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        residual = x

        out = self.conv(x)
        out = self.act(out)
        out = self.drop(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class NCPNet(nn.Module):
    """
    Non-Causal Pyramidal Context-Aware Network (NCP-Net).

    A hybrid architecture combining:
    1. A Pyramidal TCN branch for fast, resistive dynamics (Non-Causal, Wide Kernel).
    2. A High-Capacity Bidirectional LSTM branch for slow, elastic dynamics (Integral).
    3. A Deep Dense Fusion Head without global residual connections.
    """

    def __init__(self):
        super().__init__()

        input_dim = len(Config.FEATURE_COLS)

        # =====================================================================
        # Branch 1: Non-Causal Pyramidal TCN (Resistive Stream)
        # =====================================================================
        self.tcn_base_channels = Config.TCN_BASE_CHANNELS
        self.tcn_kernel_size = Config.TCN_KERNEL_SIZE
        self.tcn_layers_count = Config.TCN_LAYERS
        self.tcn_dropout = Config.TCN_DROPOUT

        # Initial projection to base channel dimension
        self.tcn_input_proj = nn.Conv1d(input_dim, self.tcn_base_channels, 1)

        self.tcn_layers = nn.ModuleList()
        current_channels = self.tcn_base_channels

        for i in range(self.tcn_layers_count):
            # Pyramidal Scaling: Double channels every layer
            out_channels = current_channels * 2
            dilation = 2**i

            block = PyramidalTCNBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=self.tcn_kernel_size,
                dilation=dilation,
                dropout=self.tcn_dropout,
            )
            self.tcn_layers.append(block)
            current_channels = out_channels

        self.tcn_output_dim = current_channels

        # =====================================================================
        # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
        # =====================================================================
        self.lstm_hidden_dim = Config.LSTM_HIDDEN_DIM
        self.lstm_layers = Config.LSTM_LAYERS
        self.lstm_dropout = Config.LSTM_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.lstm_dropout if self.lstm_layers > 1 else 0,
        )

        # Bidirectional output is 2 * hidden_dim
        self.lstm_output_dim = self.lstm_hidden_dim * 2

        # =====================================================================
        # Fusion Head: Coupled Latent Integration
        # =====================================================================
        # Concatenate TCN and LSTM outputs
        fusion_dim = self.tcn_output_dim + self.lstm_output_dim

        head_layers = []
        in_dim = fusion_dim

        # Deep Dense MLP
        for hidden_dim in Config.HEAD_HIDDEN_DIMS:
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            head_layers.append(nn.GELU())
            head_layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim

        # Final prediction layer
        head_layers.append(nn.Linear(in_dim, 1))

        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            out: Predicted pressure of shape (Batch, Seq_Len, 1)
        """
        # ---------------------------------------------------------------------
        # Branch 1: TCN Forward
        # ---------------------------------------------------------------------
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x_tcn = x.transpose(1, 2)
        x_tcn = self.tcn_input_proj(x_tcn)

        for layer in self.tcn_layers:
            x_tcn = layer(x_tcn)

        # Permute back to (Batch, Seq_Len, Channels)
        x_tcn = x_tcn.transpose(1, 2)

        # ---------------------------------------------------------------------
        # Branch 2: LSTM Forward
        # ---------------------------------------------------------------------
        # LSTM takes (Batch, Seq_Len, Input_Dim)
        x_lstm, _ = self.lstm(x)

        # ---------------------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------------------
        # Concatenate along the feature dimension
        fused = torch.cat([x_tcn, x_lstm], dim=2)

        # Pass through MLP Head
        out = self.head(fused)

        return out


# Alias for backward compatibility
DeepLSTM = NCPNet
