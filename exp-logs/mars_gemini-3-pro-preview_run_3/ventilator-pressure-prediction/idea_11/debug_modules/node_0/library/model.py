import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Generates a multiplicative gating vector based on static lung attributes (R, C).

    Formula: Output = Input * Gamma(R, C)
    """

    def __init__(self, input_dim, channels):
        super(FiLMLayer, self).__init__()
        # Project static features (R, C) to channel dimension
        self.fc = nn.Linear(input_dim, channels)

        # Initialize weights to produce gamma close to 1 initially (identity modulation)
        # This helps training stability at the start.
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 1.0)

    def forward(self, x, static_context):
        """
        Args:
            x: Feature maps from Conv1d. Shape: (Batch, Channels, Seq_Len)
            static_context: Static features (R, C). Shape: (Batch, Static_Dim)
        Returns:
            Modulated feature maps.
        """
        # Generate gamma: (Batch, Channels)
        gamma = self.fc(static_context)

        # Reshape for broadcasting: (Batch, Channels, 1)
        gamma = gamma.unsqueeze(2)

        # Apply modulation
        return x * gamma


class GatedTCNBlock(nn.Module):
    """
    A single block of the Non-Causal TCN with FiLM gating.
    Conv1d -> Activation -> FiLM -> Dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, static_dim):
        super(GatedTCNBlock, self).__init__()

        # Padding = (Kernel_Size - 1) // 2 for 'same' length with odd kernel
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="replicate",  # Replicate padding reduces edge artifacts
        )
        self.activation = nn.ELU()
        self.film = FiLMLayer(static_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static_context):
        x = self.conv(x)
        x = self.activation(x)
        x = self.film(x, static_context)
        x = self.dropout(x)
        return x


class PMNCNet(nn.Module):
    """
    Physically-Modulated Non-Causal Hybrid (PM-NC-Net).

    Structure:
    1. Input Parsing: Separates dynamic sequence from static context (R, C).
    2. Branch 1 (Resistive): TCN modulated by R, C via FiLM.
    3. Branch 2 (Elastic): High-capacity BiLSTM.
    4. Head: Concatenation -> MLP -> Pressure.
    """

    def __init__(self):
        super(PMNCNet, self).__init__()

        # ------------------------------------------------------------------
        # 1. Feature Identification
        # ------------------------------------------------------------------
        self.feature_cols = Config.FEATURE_COLS
        self.static_cols = Config.STATIC_FEATURES

        # Find indices for static features to extract them in forward pass
        self.static_indices = [self.feature_cols.index(c) for c in self.static_cols]
        self.static_dim = len(self.static_cols)

        # Input dimension for branches is total features
        input_dim = len(self.feature_cols)

        # ------------------------------------------------------------------
        # 2. Branch 1: Modulated Non-Causal TCN (Resistive Stream)
        # ------------------------------------------------------------------
        tcn_channels = Config.TCN_CHANNELS
        tcn_layers = Config.TCN_LAYERS
        kernel_size = Config.TCN_KERNEL_SIZE
        tcn_dropout = Config.TCN_DROPOUT

        self.tcn_blocks = nn.ModuleList()

        # First block projects input to channel dim
        self.tcn_blocks.append(
            GatedTCNBlock(
                input_dim, tcn_channels, kernel_size, tcn_dropout, self.static_dim
            )
        )

        # Subsequent blocks
        for _ in range(tcn_layers - 1):
            self.tcn_blocks.append(
                GatedTCNBlock(
                    tcn_channels,
                    tcn_channels,
                    kernel_size,
                    tcn_dropout,
                    self.static_dim,
                )
            )

        # ------------------------------------------------------------------
        # 3. Branch 2: High-Capacity BiLSTM (Elastic Stream)
        # ------------------------------------------------------------------
        lstm_hidden = Config.LSTM_HIDDEN_DIM
        lstm_layers = Config.LSTM_LAYERS
        lstm_dropout = Config.LSTM_DROPOUT

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # ------------------------------------------------------------------
        # 4. Fusion Head
        # ------------------------------------------------------------------
        # TCN output dim: tcn_channels
        # LSTM output dim: lstm_hidden * 2 (bidirectional)
        fusion_dim = tcn_channels + (lstm_hidden * 2)
        fc_hidden = Config.FC_HIDDEN_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fc_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ELU(),
            nn.Linear(fc_hidden // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Features)
        Returns:
            Prediction tensor of shape (Batch, Seq_Len)
        """
        # 1. Extract Static Context (R, C)
        # R and C are constant across time_step (dim 1). Take the first time step.
        # Shape: (Batch, Static_Dim)
        static_context = x[:, 0, self.static_indices]

        # 2. LSTM Branch (Elastic)
        # Input: (Batch, Seq_Len, Features)
        # Output: (Batch, Seq_Len, Hidden*2)
        lstm_out, _ = self.lstm(x)

        # 3. TCN Branch (Resistive)
        # Conv1d expects (Batch, Channels, Seq_Len). Transpose input.
        tcn_out = x.transpose(1, 2)

        for block in self.tcn_blocks:
            tcn_out = block(tcn_out, static_context)

        # Transpose back to (Batch, Seq_Len, Channels)
        tcn_out = tcn_out.transpose(1, 2)

        # 4. Fusion
        # Concatenate along feature dimension
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # Project to scalar pressure
        # Shape: (Batch, Seq_Len, 1)
        out = self.head(combined)

        # Remove last dimension to match target shape (Batch, Seq_Len)
        return out.squeeze(-1)
