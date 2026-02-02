import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedDeepBiLSTM(nn.Module):
    """
    Unified Deep Residual BiLSTM with Input Injection.
    Replaces the Wide-and-Deep architecture to favor a unified recurrent backbone
    that learns both instantaneous and temporal dynamics (Cite Lesson 00025).
    """

    def __init__(self, input_dim, config):
        super().__init__()
        self.hidden_size = config.LSTM_HIDDEN_SIZE
        self.layers = config.LSTM_LAYERS

        # Project raw inputs to latent space before injection (Cite Lesson 00028)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(config.LSTM_DROPOUT)

        for i in range(self.layers):
            # Layer 0 input: projected input (hidden_size)
            # Layer >0 input: prev_output (2*hidden) + projected input (hidden)
            # Input Injection: Concatenate projected input to hidden state (Cite Lesson 00021)
            input_size = (
                self.hidden_size
                if i == 0
                else (self.hidden_size * 2 + self.hidden_size)
            )

            self.lstm_layers.append(
                nn.LSTM(
                    input_size, self.hidden_size, batch_first=True, bidirectional=True
                )
            )
            # Layer Normalization for stability in deep RNNs (Cite Lesson 00023)
            self.layer_norms.append(nn.LayerNorm(self.hidden_size * 2))

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)

        # Project raw features to latent dimension
        x_proj = self.input_proj(x)

        h = x_proj

        for i in range(self.layers):
            if i > 0:
                # Input Injection: Concatenate projected input with previous layer output
                h = torch.cat([h, x_proj], dim=-1)

            h, _ = self.lstm_layers[i](h)
            h = self.layer_norms[i](h)

            if i < self.layers - 1:
                h = self.dropout(h)

        return self.head(h).squeeze(-1)
