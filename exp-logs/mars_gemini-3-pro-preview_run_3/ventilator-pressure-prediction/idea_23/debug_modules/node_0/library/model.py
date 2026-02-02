import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """
    Residual Dense Convolutional Block.
    Topology: Input + [Conv1D -> BN -> GELU -> Dropout -> Conv1D -> BN -> GELU -> Dropout]

    This block is designed to model local, derivative-dependent dynamics (Resistive components)
    while maintaining gradient flow via the residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()

        # Calculate padding to maintain sequence length (Same padding)
        # Assuming kernel_size is odd
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Projection for residual connection if dimensions change
        self.resize = None
        if in_channels != out_channels:
            self.resize = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        # First Conv Block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        # Second Conv Block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)

        # Residual Connection
        if self.resize is not None:
            residual = self.resize(residual)

        return out + residual


class RDHNet(nn.Module):
    """
    Residual-Dense Hybrid Network (RDH-Net).

    Architecture:
    1. Parallel Hybrid Design:
       - Branch 1: Deep Residual Dense TCN (Resistive Stream)
       - Branch 2: High-Capacity Bi-LSTM (Elastic Stream)
    2. Fusion Head:
       - Concatenation -> Wide Dense Layer -> Output
    """

    def __init__(self, config):
        super().__init__()

        # --- Branch 1: Deep Residual Dense TCN (Resistive Stream) ---
        # Models instantaneous, derivative-dependent pressure components (P ~ R * flow)
        self.cnn_layers = nn.ModuleList()
        current_dim = config.INPUT_DIM

        for out_dim in config.CONV_FILTERS:
            self.cnn_layers.append(
                ResidualConvBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    kernel_size=config.CONV_KERNEL_SIZE,
                    dropout=config.DROPOUT,
                )
            )
            current_dim = out_dim

        self.cnn_out_dim = current_dim

        # --- Branch 2: High-Capacity Bi-LSTM (Elastic Stream) ---
        # Models integral-dependent elastic pressure components (P ~ Volume / C)
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL,
        )

        lstm_out_dim = config.LSTM_HIDDEN_SIZE * (2 if config.LSTM_BIDIRECTIONAL else 1)

        # --- Fusion Head: Wide-Latent Integration ---
        # Integrates features without bottlenecks
        fusion_input_dim = self.cnn_out_dim + lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.DENSE_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.DENSE_HIDDEN_SIZE, 1),
        )

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)

        # --- CNN Path ---
        # Permute to (Batch, Features, Seq_Len) for Conv1d
        x_cnn = x.permute(0, 2, 1)

        for layer in self.cnn_layers:
            x_cnn = layer(x_cnn)

        # Permute back to (Batch, Seq_Len, Features)
        x_cnn = x_cnn.permute(0, 2, 1)

        # --- LSTM Path ---
        # LSTM expects (Batch, Seq_Len, Features)
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x)

        # --- Fusion ---
        # Concatenate along the feature dimension (dim=2)
        x_fused = torch.cat([x_cnn, x_lstm], dim=2)

        # --- Head ---
        out = self.head(x_fused)

        return out
