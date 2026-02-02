import torch
import torch.nn as nn
from library.config import Config


class DualPathInjection(nn.Module):
    """
    Implements the Dual-Path Injection block.
    Path A: Physics Context via GLU (Gated Linear Unit).
    Path B: Raw Feature Preservation via Identity.
    """

    def __init__(self, input_dim, hidden_dim):
        super(DualPathInjection, self).__init__()

        # Path A: Projects to 2*hidden for GLU, resulting in hidden_dim output
        self.context_proj = nn.Linear(input_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)

        # Path B is Identity (no parameters)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)

        # Path A: Learnable Physics Context
        context = self.context_proj(x)
        context = self.glu(context)  # Shape: (Batch, Seq, Hidden_Dim)

        # Path B: Raw Signal Preservation
        raw = x

        # Fusion: Concatenate Context and Raw Signal
        # Output dim: Hidden_Dim + Input_Dim
        payload = torch.cat([context, raw], dim=-1)

        return payload


class DP_GI_BiLSTM(nn.Module):
    """
    Dual-Path Gated-Injection BiLSTM.
    Features deep injection of the dual-path payload at every layer.
    """

    def __init__(self, config=Config):
        super(DP_GI_BiLSTM, self).__init__()

        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.LSTM_HIDDEN
        self.num_layers = config.LSTM_LAYERS
        self.bidirectional = config.BIDIRECTIONAL
        self.dropout_prob = config.DROPOUT

        # 1. Injection Block
        self.injection_block = DualPathInjection(self.input_dim, self.hidden_dim)

        # Dimension of the payload injected at every step
        self.injection_dim = self.hidden_dim + self.input_dim

        # 2. Deep Recurrent Backbone
        # We use ModuleList to allow for manual injection between layers
        self.lstm_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        num_directions = 2 if self.bidirectional else 1
        self.rnn_output_dim = self.hidden_dim * num_directions

        for i in range(self.num_layers):
            # Calculate input size for this specific layer
            if i == 0:
                # First layer receives just the injection payload
                layer_input_size = self.injection_dim
            else:
                # Subsequent layers receive: Previous Output + Injection Payload
                layer_input_size = self.rnn_output_dim + self.injection_dim

            # LSTM Layer
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )

            # Internal Regularization (LayerNorm + Dropout)
            # Applied to the output of the LSTM
            self.norms.append(nn.LayerNorm(self.rnn_output_dim))
            self.dropouts.append(nn.Dropout(self.dropout_prob))

        # 3. Prediction Head
        self.head = nn.Linear(self.rnn_output_dim, 1)

    def forward(self, x):
        """
        Forward pass.
        x: (Batch, Seq_Len, Input_Dim)
        """
        # 1. Generate Injection Payload
        # This payload serves as the "Ground Truth" signal available at all depths
        injection = self.injection_block(x)  # (Batch, Seq, Injection_Dim)

        curr_input = injection

        # 2. Iterate through Deep Layers
        for i in range(self.num_layers):
            # Pass through LSTM
            # output shape: (Batch, Seq, Hidden * Num_Directions)
            output, _ = self.lstm_layers[i](curr_input)

            # Apply Regularization to the temporal features
            output = self.norms[i](output)
            output = self.dropouts[i](output)

            # Prepare input for the next layer
            if i < self.num_layers - 1:
                # Deep Injection: Concatenate current temporal state with the static injection payload
                # Note: Injection payload does NOT have dropout applied (Deterministic Highway)
                curr_input = torch.cat([output, injection], dim=-1)
            else:
                # Final layer output flows to head
                curr_input = output

        # 3. Project to Target
        pred = self.head(curr_input)  # (Batch, Seq, 1)

        # Remove last dimension to match target shape (Batch, Seq)
        return pred.squeeze(-1)
