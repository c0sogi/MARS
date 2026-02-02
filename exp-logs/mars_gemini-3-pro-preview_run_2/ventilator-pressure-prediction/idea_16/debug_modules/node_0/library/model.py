import torch
import torch.nn as nn
from library.config import Config


class SelectiveInjectionBlock(nn.Module):
    """
    Implements the Selective-Injection mechanism.

    Structure:
    1. Fidelity Path: Identity mapping of raw features to preserve floating-point
       magnitudes of physical constants (R, C) and controls (u_in).
    2. Context Path: A Gated Linear Unit (GLU) based non-linear extraction
       projected down to a bottleneck dimension.

    Output:
    Concatenation of the Fidelity Path and the Context Path.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        # Context Path Components
        # Intermediate dimension for the GLU layer.
        # 128 is chosen to capture physics interactions efficiently.
        self.glu_hidden_dim = 128

        # Projects input to 2 * glu_hidden_dim for GLU operation (split into gates and values)
        self.fc_context = nn.Linear(input_dim, self.glu_hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)

        # Projects GLU output to bottleneck dimension
        self.fc_bottleneck = nn.Linear(self.glu_hidden_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Input_Dim)

        # 1. Fidelity Path (Identity)
        # Strictly preserves raw signals.
        fidelity = x

        # 2. Context Path (Bottlenecked GLU)
        # Extracts non-linear physics context and compresses it.
        context = self.fc_context(x)
        context = self.glu(context)
        context = self.fc_bottleneck(context)

        # 3. Fusion
        # Concatenate raw signals with compressed context
        injection_payload = torch.cat([fidelity, context], dim=-1)

        return injection_payload


class HFSI_BiLSTM(nn.Module):
    """
    High-Fidelity Selective-Injection BiLSTM Architecture.

    Key Features:
    - Selective Injection Block at the input.
    - Deep Injection: The injection payload is concatenated to the input of
      EVERY LSTM layer, not just the first one.
    - Inter-layer LayerNorm and Dropout for stability.
    - No deep front-end MLP (to prevent overfitting to instantaneous correlations).
    """

    def __init__(self, config: Config, input_dim: int):
        super().__init__()
        self.config = config

        # --- Selective Injection Block ---
        self.injection_block = SelectiveInjectionBlock(input_dim, config.BOTTLENECK_DIM)

        # Dimension of the payload (Input Features + Context Bottleneck)
        self.injection_dim = input_dim + config.BOTTLENECK_DIM

        # --- Deep Recurrent Backbone ---
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        hidden_dim = config.HIDDEN_DIM
        bidirectional = config.BIDIRECTIONAL
        num_directions = 2 if bidirectional else 1

        for i in range(config.N_LAYERS):
            # Calculate input size for this layer
            if i == 0:
                # First layer receives just the injection payload
                layer_input_size = self.injection_dim
            else:
                # Subsequent layers receive:
                # Previous Layer Output (Hidden * Directions) + Injection Payload
                layer_input_size = (hidden_dim * num_directions) + self.injection_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_dim,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )

            # Inter-layer processing (applied between layers)
            if i < config.N_LAYERS - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_dim * num_directions))
                self.dropouts.append(nn.Dropout(config.DROPOUT))

        # --- Output Head ---
        # Projects from final LSTM state to scalar pressure
        self.head = nn.Linear(hidden_dim * num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Input_Dim)

        # 1. Generate Injection Payload
        # Shape: (Batch, Seq_Len, Injection_Dim)
        injection_payload = self.injection_block(x)

        # Initialize current input with the payload
        curr_input = injection_payload

        # 2. Iterate through LSTM layers with Deep Injection
        for i in range(self.config.N_LAYERS):
            # Forward pass through LSTM layer
            # output shape: (Batch, Seq_Len, Hidden * Directions)
            output, _ = self.lstm_layers[i](curr_input)

            if i < self.config.N_LAYERS - 1:
                # Apply Inter-layer Norm and Dropout
                output = self.layer_norms[i](output)
                output = self.dropouts[i](output)

                # Deep Injection: Concatenate payload to the processed output
                # New Input Shape: (Batch, Seq_Len, Hidden*Dir + Injection_Dim)
                curr_input = torch.cat([output, injection_payload], dim=-1)
            else:
                # For the last layer, the output is the final representation
                curr_input = output

        # 3. Final Projection
        # Shape: (Batch, Seq_Len, 1)
        prediction = self.head(curr_input)

        return prediction
