import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception-style).
    Extracts features using multiple kernel sizes and concatenates them.
    """

    def __init__(self, input_dim, hidden_dim, kernel_sizes):
        super().__init__()
        # Calculate channel depth for each kernel branch
        num_kernels = len(kernel_sizes)
        branch_channels = hidden_dim // num_kernels

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        # Calculate the actual concatenated dimension
        concat_dim = branch_channels * num_kernels

        # Projection to ensure output matches hidden_dim exactly
        self.projection = nn.Linear(concat_dim, hidden_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        outputs = [conv(x) for conv in self.convs]

        # Concatenate along channel dimension
        x = torch.cat(outputs, dim=1)

        # Transpose back: (batch, seq_len, concat_dim)
        x = x.transpose(1, 2)

        # Project to hidden_dim
        x = self.projection(x)
        return x


class CompositeBlock(nn.Module):
    """
    High-Capacity Composite Block.
    Consists of:
    1. Context-Injected Temporal Mixer (Bi-LSTM) with Physics Injection
    2. Pointwise Channel Mixer (FFN)
    3. Pure Additive Residuals (No Normalization)
    """

    def __init__(self, input_dim, hidden_dim, physics_dim, dropout=0.1):
        super().__init__()

        # 1. Context-Injected Temporal Mixer
        # Input: Previous Hidden + Physics Features
        self.lstm_input_dim = input_dim + physics_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # 2. Pointwise Channel Mixer (FFN)
        # Expansion factor of 4 for high capacity
        ffn_dim = hidden_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim)
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, physics_features):
        # x: (batch, seq_len, hidden_dim)
        # physics_features: (batch, seq_len, physics_dim)

        # --- Temporal Mixing ---
        # Inject physics features
        lstm_input = torch.cat([x, physics_features], dim=-1)

        # LSTM processing
        lstm_out, _ = self.lstm(lstm_input)

        # Additive Residual (No Norm)
        x = x + self.lstm_dropout(lstm_out)

        # --- Channel Mixing ---
        # FFN processing
        ffn_out = self.ffn(x)

        # Additive Residual (No Norm)
        x = x + self.ffn_dropout(ffn_out)

        return x


class VentilatorModel(nn.Module):
    """
    High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM-FFN.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Identify indices of physics features in the input tensor
        # We look for R, C, and their interaction terms
        target_physics_feats = ["R", "C", "R_u_in", "area_div_C"]
        self.physics_indices = []

        for feat in target_physics_feats:
            if feat in config.INPUT_FEATURES:
                self.physics_indices.append(config.INPUT_FEATURES.index(feat))

        self.physics_dim = len(self.physics_indices)

        # --- Architecture ---

        # Stem: Multi-Scale CNN
        self.stem = MultiScaleCNN(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_SIZE,
            kernel_sizes=config.CNN_KERNEL_SIZES,
        )

        # Backbone: Stack of Composite Blocks
        self.layers = nn.ModuleList()
        for _ in range(config.NUM_LAYERS):
            self.layers.append(
                CompositeBlock(
                    input_dim=config.HIDDEN_SIZE,
                    hidden_dim=config.HIDDEN_SIZE,
                    physics_dim=self.physics_dim,
                    dropout=config.DROPOUT,
                )
            )

        # Heads
        self.aux_head = nn.Linear(config.HIDDEN_SIZE, 1)
        self.final_head = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            final_pred: (batch, seq_len, 1)
            aux_pred: (batch, seq_len, 1) or None
        """
        # Extract Physics Features for Injection
        if self.physics_dim > 0:
            physics_features = x[:, :, self.physics_indices]
        else:
            # Fallback if no physics features found (should not happen with correct config)
            physics_features = torch.empty(x.size(0), x.size(1), 0, device=x.device)

        # Pass through Stem
        h = self.stem(x)

        aux_out = None

        # Pass through Composite Blocks
        for i, layer in enumerate(self.layers):
            h = layer(h, physics_features)

            # Deep Supervision: Auxiliary Head after Block 2 (index 1)
            if i == 1:
                aux_out = self.aux_head(h)

        # Final Prediction
        final_out = self.final_head(h)

        return final_out, aux_out
