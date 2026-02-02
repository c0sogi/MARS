import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepStream(nn.Module):
    """
    The 'Deep' component: A BiLSTM with Input Injection.
    Captures temporal dynamics and memory (e.g., Elastic Pressure, Integral accumulation).
    """

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.layers = layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Layer 1: Standard BiLSTM taking raw features
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim * 2))

        # Subsequent Layers: Input Injection (Previous Hidden + Raw Features)
        for _ in range(1, layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=(hidden_dim * 2) + input_dim,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * 2))

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        h = x

        for i in range(self.layers):
            if i > 0:
                # Input Injection: Concatenate raw input x with output of previous layer
                h = torch.cat([h, x], dim=-1)

            h, _ = self.lstm_layers[i](h)
            h = self.layer_norms[i](h)

            # Apply dropout between layers (but not after the last one)
            if i < self.layers - 1:
                h = self.dropout(h)

        return h


class UnifiedBiLSTM(nn.Module):
    """
    Unified Deep Residual BiLSTM with Input Injection.
    Replaces branching architecture with a single powerful recurrent backbone.
    Cite Lesson 00025.
    """

    def __init__(self, input_dim, config):
        super().__init__()

        # Deep Stream Configuration (The Unified Backbone)
        self.backbone = DeepStream(
            input_dim=input_dim,
            hidden_dim=config.LSTM_HIDDEN_SIZE,
            layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT,
        )

        # Regression Head
        # Projects BiLSTM Output (Hidden * 2) to scalar Pressure
        backbone_out_dim = config.LSTM_HIDDEN_SIZE * 2

        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, config.LSTM_HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(config.LSTM_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # 1. Unified Recurrent Processing
        features = self.backbone(x)

        # 2. Prediction
        pred = self.head(features)

        return pred.squeeze(-1)
