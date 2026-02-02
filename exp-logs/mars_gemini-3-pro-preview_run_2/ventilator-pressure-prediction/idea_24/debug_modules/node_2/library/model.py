import torch
import torch.nn as nn
from library.config import Config


class InjectionBlock(nn.Module):
    """
    Implements the 'Wide-Bandwidth Injection Mechanism'.

    Consists of two parallel paths:
    1. Path A (Explicit Fidelity): Identity mapping of raw inputs.
    2. Path B (Wide Context): A Monolithic Gated Linear Unit (GLU) processing all features.

    The outputs are concatenated to form a high-fidelity payload for deep injection.
    """

    def __init__(self, input_dim, glu_width):
        super(InjectionBlock, self).__init__()
        # Path B: Wide Monolithic GLU
        # Maps input -> 2 * width -> GLU -> width
        self.glu_path = nn.Sequential(
            nn.Linear(input_dim, 2 * glu_width), nn.GLU(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features. Shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Injection payload. Shape (Batch, Seq_Len, Input_Dim + Glu_Width)
        """
        # Path A: Identity (Preserve raw physics values)
        path_a = x

        # Path B: Wide Context (Learn non-linear interactions)
        path_b = self.glu_path(x)

        # Fusion: Concatenate without Dropout
        return torch.cat([path_a, path_b], dim=-1)


class DeepBiLSTM(nn.Module):
    """
    Implements the 'Uncompressed Physics-Context Deep-Injection BiLSTM'.

    Features:
    - 4-Layer Bidirectional LSTM Backbone (512 hidden units).
    - Deep Injection: The InjectionBlock payload is concatenated to the input of EVERY layer.
    - Layer Normalization and Dropout between layers.
    - No Output Skip-Connections (Forces temporal state interaction).
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        glu_width=Config.INJECTION_GLU_WIDTH,
        dropout=Config.DROPOUT,
    ):
        super(DeepBiLSTM, self).__init__()

        self.hidden_size = hidden_size

        # 1. Injection Block
        self.injection_block = InjectionBlock(input_dim, glu_width)
        injection_dim = input_dim + glu_width

        # 2. Deep Recurrent Backbone
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            # Calculate input size for the current LSTM layer
            # Layer 0: Input is just the Injection Payload
            # Layer >0: Input is (Previous Layer Output) + (Injection Payload)
            if i == 0:
                layer_input_size = injection_dim
            else:
                # Bidirectional output from previous layer = 2 * hidden_size
                layer_input_size = (2 * hidden_size) + injection_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Layer Norm scales the output features (2 * hidden_size)
            self.layer_norms.append(nn.LayerNorm(2 * hidden_size))
            self.dropouts.append(nn.Dropout(dropout))

        # 3. Regression Head
        # Maps final LSTM state to scalar pressure
        self.head = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features. Shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Predicted pressure. Shape (Batch, Seq_Len)
        """
        # Generate the Wide-Bandwidth Payload
        # Shape: (Batch, Seq_Len, Input_Dim + Glu_Width)
        payload = self.injection_block(x)

        curr_input = payload

        # Iterate through LSTM stack
        for i, (lstm, ln, drop) in enumerate(
            zip(self.lstm_layers, self.layer_norms, self.dropouts)
        ):
            # Forward pass through LSTM
            lstm_out, _ = lstm(curr_input)  # Shape: (Batch, Seq, 2*Hidden)

            # Apply Stabilization (LN + Dropout)
            lstm_out = ln(lstm_out)
            lstm_out = drop(lstm_out)

            # Prepare input for the next layer (Deep Injection)
            # If not the last layer, concatenate the payload to the current output
            if i < len(self.lstm_layers) - 1:
                curr_input = torch.cat([lstm_out, payload], dim=-1)
            else:
                final_out = lstm_out

        # Project to scalar prediction
        pressure = self.head(final_out)  # Shape: (Batch, Seq, 1)

        # Remove last dimension to match target shape
        return pressure.squeeze(-1)
