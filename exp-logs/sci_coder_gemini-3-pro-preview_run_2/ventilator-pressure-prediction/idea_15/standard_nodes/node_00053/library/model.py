import torch
import torch.nn as nn
from library.config import Config


class GatedResidualBlock(nn.Module):
    """
    A Gated Residual Block designed for the Front-Loaded Context Extractor.
    Structure: Linear -> LayerNorm -> GLU -> Dropout -> Residual Add
    """

    def __init__(self, dim, dropout=0.1):
        super(GatedResidualBlock, self).__init__()
        # Project to 2*dim because GLU halves the dimension
        self.linear = nn.Linear(dim, dim * 2)
        self.ln = nn.LayerNorm(dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.ln(out)
        out = self.glu(out)
        out = self.dropout(out)
        return residual + out


class DFLB_BiLSTM(nn.Module):
    """
    Deep Front-Loaded Bottlenecked BiLSTM (DFLB-BiLSTM) Architecture.

    Features:
    - Front-Loaded Context Extractor: Deep Gated Residual MLP for instantaneous physics.
    - Bottlenecked Injection: Compresses context to preserve raw signal fidelity.
    - Deep Injection: Concatenates payload to the input of every LSTM layer.
    """

    def __init__(self, input_dim):
        super(DFLB_BiLSTM, self).__init__()

        # ==========================================
        # 1. Front-Loaded Context Extractor
        # ==========================================
        # Initial projection to hidden dimension
        self.front_end_input_proj = nn.Linear(input_dim, Config.FRONT_END_DIM)

        # Stack of Gated Residual Blocks
        self.front_end_blocks = nn.ModuleList(
            [
                GatedResidualBlock(Config.FRONT_END_DIM, dropout=0.1)
                for _ in range(Config.FRONT_END_LAYERS)
            ]
        )

        # ==========================================
        # 2. Bottleneck Projection
        # ==========================================
        # Compresses the high-dimensional context
        self.bottleneck_proj = nn.Linear(Config.FRONT_END_DIM, Config.BOTTLENECK_DIM)

        # ==========================================
        # 3. Deep Recurrent Backbone
        # ==========================================
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # The payload consists of the Raw Input (Path A) and Compressed Context (Path B)
        payload_dim = input_dim + Config.BOTTLENECK_DIM

        hidden_dim = Config.LSTM_DIM
        num_directions = 2 if Config.BIDIRECTIONAL else 1
        lstm_output_dim = hidden_dim * num_directions

        for i in range(Config.LSTM_LAYERS):
            # Determine input size for this layer
            # Layer 0: Input is just the Payload
            # Layer k>0: Input is (Previous Layer Output) + (Payload)
            if i == 0:
                layer_input_size = payload_dim
            else:
                layer_input_size = lstm_output_dim + payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=Config.BIDIRECTIONAL,
                )
            )

            # Inter-layer processing
            self.layer_norms.append(nn.LayerNorm(lstm_output_dim))
            # Standard inter-layer dropout
            self.dropouts.append(nn.Dropout(0.1))

        # ==========================================
        # 4. Head
        # ==========================================
        if Config.HEAD_DROPOUT > 0:
            self.head_dropout = nn.Dropout(Config.HEAD_DROPOUT)
        else:
            self.head_dropout = nn.Identity()

        self.head = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len)
        """
        # --- 1. Front-End Processing ---
        # Project raw input to front-end dimension
        ctx = self.front_end_input_proj(x)

        # Pass through Deep Gated Residual MLP
        for block in self.front_end_blocks:
            ctx = block(ctx)

        # --- 2. Bottleneck Compression ---
        ctx_compressed = self.bottleneck_proj(ctx)

        # --- 3. Payload Construction ---
        # Path A: Raw Signal Fidelity (x)
        # Path B: Compressed Context (ctx_compressed)
        payload = torch.cat([x, ctx_compressed], dim=-1)

        # --- 4. Deep Recurrent Injection ---
        curr_input = payload
        final_output = None

        for i, lstm in enumerate(self.lstm_layers):
            # LSTM Forward Pass
            lstm_out, _ = lstm(curr_input)

            # Apply LayerNorm and Dropout
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.dropouts[i](lstm_out)

            # Prepare input for the next layer
            if i < len(self.lstm_layers) - 1:
                # Inject Payload: Concat(Current Output, Payload)
                curr_input = torch.cat([lstm_out, payload], dim=-1)
            else:
                final_output = lstm_out

        # --- 5. Head ---
        final_output = self.head_dropout(final_output)
        pred = self.head(final_output)

        # Remove last dimension to match target shape (Batch, Seq_Len)
        return pred.squeeze(-1)
