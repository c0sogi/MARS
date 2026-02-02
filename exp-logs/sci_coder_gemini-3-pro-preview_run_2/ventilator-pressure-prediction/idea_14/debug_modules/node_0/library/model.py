import torch
import torch.nn as nn
from library.config import Config


class GatedResidualBlock(nn.Module):
    """
    A non-linear feature extraction unit consisting of:
    Linear -> LayerNorm -> GLU -> Dropout -> Residual Add
    """

    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        # GLU halves the dimension, so Linear must output 2 * output_dim
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.ln = nn.LayerNorm(output_dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if dimensions mismatch
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)

        # Linear -> LayerNorm -> GLU -> Dropout
        x = self.fc(x)
        x = self.ln(x)
        x = self.glu(x)
        x = self.dropout(x)

        return x + residual


class DFL_GI_BiLSTM(nn.Module):
    """
    Deep Front-Loaded Gated-Injection BiLSTM (DFL-GI-BiLSTM)

    Architecture:
    1. Deep Front-End: Stack of GatedResidualBlocks to extract physics context.
    2. Bottleneck: Compresses context.
    3. Injection: Concatenates Raw Input + Context.
    4. Backbone: 4-layer BiLSTM with Deep Injection (Payload injected at every layer).
    """

    def __init__(self):
        super().__init__()

        # Configuration
        input_dim = Config.INPUT_DIM
        hidden_dim = Config.LSTM_HIDDEN_DIM
        bottleneck_dim = Config.BOTTLENECK_DIM
        num_layers = Config.LSTM_LAYERS
        dropout_p = Config.DROPOUT
        fe_layers = Config.FRONT_END_LAYERS

        # --- 1. Deep Front-End ---
        # Initial projection to hidden dimension
        self.fe_input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Gated Residual Blocks
        # We use hidden_dim as the working dimension for the front-end
        self.fe_blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, hidden_dim, dropout_p)
                for _ in range(fe_layers)
            ]
        )

        # Bottleneck Projection (No Dropout here strictly)
        self.bottleneck_proj = nn.Linear(hidden_dim, bottleneck_dim)

        # --- 2. Deep Recurrent Backbone ---
        self.lstm_layers = nn.ModuleList()
        self.lstm_lns = nn.ModuleList()
        self.lstm_drops = nn.ModuleList()

        # Injection Payload = Raw Input + Bottleneck Context
        payload_dim = input_dim + bottleneck_dim

        for i in range(num_layers):
            # Determine input size for this LSTM layer
            if i == 0:
                # First layer receives just the payload
                layer_input_dim = payload_dim
            else:
                # Subsequent layers receive: Previous Output (Bidirectional) + Payload
                layer_input_dim = (hidden_dim * 2) + payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Inter-layer processing (LN + Drop)
            # Not applied after the final layer
            if i < num_layers - 1:
                self.lstm_lns.append(nn.LayerNorm(hidden_dim * 2))
                self.lstm_drops.append(nn.Dropout(dropout_p))

        # --- 3. Regression Head ---
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            Predictions of shape (Batch, Seq_Len)
        """
        # --- Front-End Processing ---
        # Project and extract features
        fe = self.fe_input_proj(x)
        for block in self.fe_blocks:
            fe = block(fe)

        # Create Physics Context (Bottleneck)
        context = self.bottleneck_proj(fe)

        # --- Injection Payload Construction ---
        # Concatenate Raw Input and Physics Context
        payload = torch.cat([x, context], dim=-1)

        # --- Recurrent Backbone ---
        curr_input = payload

        for i, lstm in enumerate(self.lstm_layers):
            # For layers > 0, inject payload by concatenating with previous output
            if i > 0:
                curr_input = torch.cat([curr_input, payload], dim=-1)

            # LSTM Forward
            lstm_out, _ = lstm(curr_input)

            # Apply Inter-layer Norm and Dropout
            if i < len(self.lstm_layers) - 1:
                lstm_out = self.lstm_lns[i](lstm_out)
                lstm_out = self.lstm_drops[i](lstm_out)

            # Set input for next layer
            curr_input = lstm_out

        # --- Head ---
        # curr_input is the output of the last LSTM layer
        pred = self.head(curr_input)

        # Remove last dimension to match target shape (Batch, Seq_Len)
        return pred.squeeze(-1)
