import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """
    A residual block for the Wide Stream MLP.
    Structure: Input -> Linear -> GELU -> Dropout -> Linear -> Dropout -> Add -> Output
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class WideStream(nn.Module):
    """
    The 'Wide' component: A Residual MLP processing time steps independently.
    Captures instantaneous relationships (e.g., Resistive Pressure).
    """

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(layers)]
        )

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        return x


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


class WideDeepBiLSTM(nn.Module):
    """
    Wide-and-Deep Physics-Informed Architecture.
    Fuses the Deep Recurrent Stream and Wide Instantaneous Stream.
    """

    def __init__(self, input_dim, config):
        super().__init__()

        # Deep Stream Configuration
        self.deep_stream = DeepStream(
            input_dim=input_dim,
            hidden_dim=config.LSTM_HIDDEN_SIZE,
            layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT,
        )

        # Wide Stream Configuration
        self.wide_stream = WideStream(
            input_dim=input_dim,
            hidden_dim=config.MLP_HIDDEN_SIZE,
            layers=config.MLP_LAYERS,
            dropout=config.MLP_DROPOUT,
        )

        # Fusion Head
        # Concatenates (BiLSTM Output) and (MLP Output)
        deep_out_dim = config.LSTM_HIDDEN_SIZE * 2
        wide_out_dim = config.MLP_HIDDEN_SIZE
        fusion_dim = deep_out_dim + wide_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # 1. Deep Recurrent Stream
        deep_out = self.deep_stream(x)

        # 2. Wide Instantaneous Stream
        wide_out = self.wide_stream(x)

        # 3. Fusion
        combined = torch.cat([deep_out, wide_out], dim=-1)
        pred = self.head(combined)

        return pred.squeeze(-1)
